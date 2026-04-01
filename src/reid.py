"""
reid.py — Cross-Camera Vehicle Re-Identification (Qdrant)
==========================================================
Match strategy:
  1. Plate match     — exact string, confidence = 1.0
  2. Qdrant search   — cosine similarity on 512-dim ResNet18 vector
  3. New vehicle     — insert into Qdrant + in-memory registry

Qdrant collection: one point per GlobalVehicle (not per sighting).
The point vector is always updated to the latest sighting's embedding.
Payload stores global_id + plate for fast lookup after search.

In-memory GlobalVehicle registry is kept for sightings history,
movement_summary, and graph queries — Qdrant handles only the
similarity search part.

Config:
    QDRANT_IN_MEMORY = True   → qdrant_client.QdrantClient(":memory:")
    QDRANT_IN_MEMORY = False  → QdrantClient(host, port)  (persistent server)
"""

import logging
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, MatchValue,
    UpdateStatus,
)

import config
from embedder import VehicleEmbedder, VehicleEmbedding

logger = logging.getLogger("ReID")


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class CameraSighting:
    camera_id:  str
    track_id:   int
    timestamp:  datetime
    bbox:       tuple
    plate:      Optional[str]  = None
    confidence: float          = 0.0
    location:   Optional[str]  = None


@dataclass
class GlobalVehicle:
    global_id:     int
    qdrant_id:     str                        # UUID used as Qdrant point id
    sightings:     list[CameraSighting]       = field(default_factory=list)
    plate:         Optional[str]              = None
    is_motorcycle: bool                       = False
    embedding:     Optional[VehicleEmbedding] = None
    local_tracks:  dict                       = field(default_factory=lambda: defaultdict(list))
    first_seen:    Optional[datetime]         = None
    last_seen:     Optional[datetime]         = None

    def add_sighting(self, s: CameraSighting) -> None:
        self.sightings.append(s)
        self.local_tracks[s.camera_id].append(s.track_id)
        if self.plate is None and s.plate:
            self.plate = s.plate
            logger.info(f"[ReID] GV#{self.global_id} — plate confirmed: {self.plate}")
        self.first_seen = self.first_seen or s.timestamp
        self.last_seen  = s.timestamp

    @property
    def camera_count(self) -> int:
        return len(set(s.camera_id for s in self.sightings))

    @property
    def movement_summary(self) -> list[dict]:
        emb   = self.embedding
        attrs = {}
        if emb:
            attrs = {
                "color":          emb.dominant_color,
                "helmet_present": emb.helmet_present,
                "helmet_color":   emb.helmet_color,
                "clothing_color": emb.clothing_color,
            }
        return [
            {
                "camera_id": s.camera_id,
                "location":  s.location or s.camera_id,
                "timestamp": s.timestamp.isoformat(),
                "plate":     s.plate,
                "track_id":  s.track_id,
                **attrs,
            }
            for s in sorted(self.sightings, key=lambda x: x.timestamp)
        ]


# ══════════════════════════════════════════════════════════════════
# RE-ID ENGINE
# ══════════════════════════════════════════════════════════════════

class ReIDEngine:

    def __init__(self):
        self._embedder   = VehicleEmbedder()
        self._registry:  list[GlobalVehicle] = []
        self._id_counter = 0

        # Qdrant client
        if config.QDRANT_IN_MEMORY:
            self._qdrant = QdrantClient(":memory:")
            logger.info("Qdrant: in-memory mode.")
        else:
            self._qdrant = QdrantClient(
                host=config.QDRANT_HOST, port=config.QDRANT_PORT
            )
            logger.info(f"Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")

        self._ensure_collection()
        logger.info("ReID Engine initialised.")

    # ── Qdrant setup ───────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if config.QDRANT_COLLECTION not in existing:
            self._qdrant.create_collection(
                collection_name = config.QDRANT_COLLECTION,
                vectors_config  = VectorParams(
                    size     = config.EMBED_DIM,
                    distance = Distance.COSINE,
                ),
            )
            logger.info(f"Qdrant collection '{config.QDRANT_COLLECTION}' created.")

    # ── Main match API ─────────────────────────────────────────────

    def match_or_create(
        self,
        camera_id:     str,
        track_id:      int,
        crop:          object,
        plate:         Optional[str]      = None,
        is_motorcycle: bool               = False,
        timestamp:     Optional[datetime] = None,
        bbox:          tuple              = (0, 0, 0, 0),
        plate_chars:   str                = "",
    ) -> GlobalVehicle:
        if timestamp is None:
            timestamp = datetime.now()

        embedding = self._embedder.extract(
            crop,
            vehicle_type = "motorcycle" if is_motorcycle else "vehicle",
            plate_chars  = plate_chars,
        )

        location = config.CAMERA_LOCATIONS.get(camera_id, camera_id)
        sighting = CameraSighting(
            camera_id  = camera_id,
            track_id   = track_id,
            timestamp  = timestamp,
            bbox       = bbox,
            plate      = plate,
            location   = location,
        )

        # 1. Plate match (no Qdrant needed)
        if plate:
            gv = self._match_by_plate(plate)
            if gv:
                sighting.confidence = 1.0
                gv.add_sighting(sighting)
                if embedding:
                    self._upsert_vector(gv, embedding)
                logger.debug(f"[ReID] Plate match → GV#{gv.global_id} ({plate})")
                return gv

        # 2. Qdrant vector search
        if embedding is not None:
            gv, score = self._search_qdrant(embedding, camera_id, track_id)
            if gv and score >= config.REID_THRESHOLD:
                sighting.confidence = score
                gv.add_sighting(sighting)
                self._upsert_vector(gv, embedding)
                logger.debug(f"[ReID] Vector match → GV#{gv.global_id} score={score:.3f}")
                return gv

        # 3. New vehicle
        gv = self._create(plate, is_motorcycle, embedding)
        sighting.confidence = 1.0
        gv.add_sighting(sighting)
        if embedding:
            self._insert_vector(gv, embedding)
        logger.info(f"[ReID] New GV#{gv.global_id} plate={plate} moto={is_motorcycle}")
        return gv

    # ── Qdrant operations ──────────────────────────────────────────

    def _search_qdrant(
        self,
        embedding: VehicleEmbedding,
        camera_id: str,
        track_id:  int,
    ) -> tuple[Optional[GlobalVehicle], float]:
        try:
            result = self._qdrant.query_points(
                collection_name = config.QDRANT_COLLECTION,
                query           = embedding.vector.tolist(),
                limit           = 5,
                score_threshold = config.REID_THRESHOLD,
                with_payload    = True,
            )
            hits = result.points
        except Exception as e:
            logger.warning(f"Qdrant search failed: {e}")
            return None, 0.0

        for hit in hits:
            gv_id = hit.payload.get("global_id")
            gv    = self.get_vehicle(gv_id)
            if gv is None:
                continue
            # Skip if this camera+track already belongs to this vehicle
            if track_id in gv.local_tracks.get(camera_id, []):
                continue
            return gv, float(hit.score)

        return None, 0.0

    def _insert_vector(self, gv: GlobalVehicle, emb: VehicleEmbedding) -> None:
        try:
            self._qdrant.upsert(
                collection_name = config.QDRANT_COLLECTION,
                points = [PointStruct(
                    id      = gv.qdrant_id,
                    vector  = emb.vector.tolist(),
                    payload = {
                        "global_id":      gv.global_id,
                        "plate":          gv.plate,
                        "dominant_color": emb.dominant_color,
                        "helmet_present": emb.helmet_present,
                        "vehicle_type":   emb.vehicle_type,
                    },
                )],
            )
            gv.embedding = emb
        except Exception as e:
            logger.warning(f"Qdrant insert failed for GV#{gv.global_id}: {e}")

    def _upsert_vector(self, gv: GlobalVehicle, emb: VehicleEmbedding) -> None:
        """Update vector + payload to latest sighting."""
        gv.embedding = emb
        self._insert_vector(gv, emb)

    # ── In-memory helpers ──────────────────────────────────────────

    def _match_by_plate(self, plate: str) -> Optional[GlobalVehicle]:
        return next((gv for gv in self._registry if gv.plate == plate), None)

    def _create(
        self,
        plate:         Optional[str],
        is_motorcycle: bool,
        embedding:     Optional[VehicleEmbedding],
    ) -> GlobalVehicle:
        self._id_counter += 1
        gv = GlobalVehicle(
            global_id     = self._id_counter,
            qdrant_id     = str(uuid.uuid4()),
            plate         = plate,
            is_motorcycle = is_motorcycle,
            embedding     = embedding,
        )
        self._registry.append(gv)
        return gv

    # ── Registry queries ───────────────────────────────────────────

    def get_vehicle(self, global_id: int) -> Optional[GlobalVehicle]:
        return next((gv for gv in self._registry if gv.global_id == global_id), None)

    def get_all_vehicles(self) -> list[GlobalVehicle]:
        return self._registry

    def get_motorcycles(self) -> list[GlobalVehicle]:
        return [gv for gv in self._registry if gv.is_motorcycle]

    def get_multi_camera_vehicles(self) -> list[GlobalVehicle]:
        return [gv for gv in self._registry if gv.camera_count > 1]

    def summary(self) -> dict:
        return {
            "total_vehicles":        len(self._registry),
            "motorcycles":           len(self.get_motorcycles()),
            "multi_camera_vehicles": len(self.get_multi_camera_vehicles()),
        }