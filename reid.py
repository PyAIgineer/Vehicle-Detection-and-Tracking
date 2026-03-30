"""
reid.py — Cross-Camera Vehicle Re-Identification
=================================================
Two matching strategies, automatically fused:

  1. PLATE-BASED ReID  — exact match from LPR pipeline
                         (high confidence, used when plate is readable)

  2. VISUAL ReID       — colour histogram + appearance embedding
                         (used when plate is absent/partial)

Outputs:
    Global vehicle identity that persists across cameras,
    linking camera-local track_ids under one GlobalVehicle.
"""

import cv2
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import config

logger = logging.getLogger("ReID")


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class VehicleAppearance:
    """Visual fingerprint of a vehicle crop."""
    color_hist:   np.ndarray    # flattened HSV histogram
    embedding:    Optional[np.ndarray] = None   # deep ReID embedding (if model loaded)


@dataclass
class CameraSighting:
    camera_id:  str
    track_id:   int
    timestamp:  datetime
    bbox:       tuple
    plate:      Optional[str] = None
    confidence: float = 0.0
    location:   Optional[str] = None


@dataclass
class GlobalVehicle:
    """
    A vehicle identity that spans multiple cameras.
    All camera-local tracks that match this identity are grouped here.
    """
    global_id:    int
    sightings:    list[CameraSighting] = field(default_factory=list)
    plate:        Optional[str]        = None     # confirmed plate (if any)
    is_motorcycle:bool                 = False
    appearance:   Optional[VehicleAppearance] = None

    # Camera-local track mapping: {camera_id: [track_id, ...]}
    local_tracks: dict = field(default_factory=lambda: defaultdict(list))

    first_seen:   Optional[datetime] = None
    last_seen:    Optional[datetime] = None

    def add_sighting(self, sighting: CameraSighting) -> None:
        self.sightings.append(sighting)
        self.local_tracks[sighting.camera_id].append(sighting.track_id)
        if self.plate is None and sighting.plate:
            self.plate = sighting.plate
            logger.info(f"[ReID] GlobalVehicle #{self.global_id} — plate confirmed: {self.plate}")
        if self.first_seen is None:
            self.first_seen = sighting.timestamp
        self.last_seen = sighting.timestamp

    @property
    def camera_count(self) -> int:
        return len(set(s.camera_id for s in self.sightings))

    @property
    def movement_summary(self) -> list[dict]:
        """Ordered list of camera transitions."""
        return [
            {
                "camera_id": s.camera_id,
                "location":  s.location or s.camera_id,
                "timestamp": s.timestamp.isoformat(),
                "plate":     s.plate,
                "track_id":  s.track_id,
            }
            for s in sorted(self.sightings, key=lambda x: x.timestamp)
        ]


# ══════════════════════════════════════════════════════════════════
# VISUAL FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

class VisualExtractor:
    """
    Extracts colour histogram features from vehicle crops.
    Optionally uses a deep ReID model (torchreid / OSNet) if available.
    """

    def __init__(self):
        self._reid_model = None
        self._try_load_reid_model()

    def _try_load_reid_model(self) -> None:
        try:
            import torchreid
            self._reid_model = torchreid.models.build_model(
                name=config.REID_MODEL,
                num_classes=1,
                pretrained=True,
            )
            self._reid_model.eval()
            logger.info(f"Deep ReID model loaded: {config.REID_MODEL}")
        except Exception as e:
            logger.info(f"Deep ReID not available ({e}). Using colour histogram only.")
            self._reid_model = None

    def extract(self, crop: np.ndarray) -> Optional[VehicleAppearance]:
        if crop is None or crop.size == 0:
            return None

        hist = self._colour_histogram(crop)

        embedding = None
        if self._reid_model is not None:
            try:
                embedding = self._deep_embedding(crop)
            except Exception as e:
                logger.debug(f"Deep embedding failed: {e}")

        return VehicleAppearance(color_hist=hist, embedding=embedding)

    def _colour_histogram(self, crop: np.ndarray) -> np.ndarray:
        """HSV colour histogram normalised to unit vector."""
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bins = config.COLOR_HIST_BINS
        hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _deep_embedding(self, crop: np.ndarray) -> np.ndarray:
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            feat = self._reid_model(tensor)
        feat = feat.squeeze().cpu().numpy()
        return feat / (np.linalg.norm(feat) + 1e-8)


# ══════════════════════════════════════════════════════════════════
# SIMILARITY SCORING
# ══════════════════════════════════════════════════════════════════

def _hist_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance → similarity score (0–1)."""
    dist = cv2.compareHist(
        h1.reshape(-1, 1).astype(np.float32),
        h2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA
    )
    return float(max(0.0, 1.0 - dist))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compute_appearance_similarity(ap1: VehicleAppearance, ap2: VehicleAppearance) -> float:
    """Fuse histogram and deep embedding similarities."""
    hist_sim = _hist_similarity(ap1.color_hist, ap2.color_hist)

    if ap1.embedding is not None and ap2.embedding is not None:
        emb_sim = _cosine_similarity(ap1.embedding, ap2.embedding)
        return 0.35 * hist_sim + 0.65 * emb_sim
    return hist_sim


# ══════════════════════════════════════════════════════════════════
# RE-ID ENGINE
# ══════════════════════════════════════════════════════════════════

class ReIDEngine:
    """
    Maintains the global vehicle registry and matches new sightings
    to existing global vehicles.
    """

    def __init__(self):
        self._extractor  = VisualExtractor()
        self._registry:  list[GlobalVehicle] = []
        self._id_counter = 0
        logger.info("ReID Engine initialised.")

    # ── Main match API ─────────────────────────────────────────────

    def match_or_create(
        self,
        camera_id:    str,
        track_id:     int,
        crop:         np.ndarray,
        plate:        Optional[str] = None,
        is_motorcycle:bool = False,
        timestamp:    Optional[datetime] = None,
        bbox:         tuple = (0, 0, 0, 0),
    ) -> GlobalVehicle:
        """
        Given a detection, find the matching GlobalVehicle or create a new one.

        Priority:
          1. Plate match (exact)
          2. Visual appearance match
          3. New global vehicle
        """
        if timestamp is None:
            timestamp = datetime.now()

        appearance = self._extractor.extract(crop)
        location   = config.CAMERA_LOCATIONS.get(camera_id, camera_id)

        sighting = CameraSighting(
            camera_id   = camera_id,
            track_id    = track_id,
            timestamp   = timestamp,
            bbox        = bbox,
            plate       = plate,
            confidence  = 0.0,
            location    = location,
        )

        # ── Strategy 1: plate match ────────────────────────────────
        if plate:
            matched = self._match_by_plate(plate)
            if matched:
                sighting.confidence = 1.0
                matched.add_sighting(sighting)
                if appearance:
                    matched.appearance = appearance   # update appearance
                logger.debug(f"[ReID] Plate match → GlobalVehicle #{matched.global_id}  ({plate})")
                return matched

        # ── Strategy 2: visual match ───────────────────────────────
        if appearance:
            matched, score = self._match_by_appearance(appearance, camera_id, track_id)
            if matched and score >= config.REID_THRESHOLD:
                sighting.confidence = score
                matched.add_sighting(sighting)
                matched.appearance = appearance
                logger.debug(
                    f"[ReID] Visual match → GlobalVehicle #{matched.global_id}  "
                    f"(score={score:.3f})"
                )
                return matched

        # ── Strategy 3: new global vehicle ─────────────────────────
        gv = self._create(plate, is_motorcycle, appearance)
        sighting.confidence = 1.0
        gv.add_sighting(sighting)
        logger.info(f"[ReID] New GlobalVehicle #{gv.global_id}  plate={plate}  moto={is_motorcycle}")
        return gv

    # ── Internal match helpers ─────────────────────────────────────

    def _match_by_plate(self, plate: str) -> Optional[GlobalVehicle]:
        for gv in self._registry:
            if gv.plate == plate:
                return gv
        return None

    def _match_by_appearance(
        self,
        appearance: VehicleAppearance,
        camera_id:  str,
        track_id:   int,
    ) -> tuple[Optional[GlobalVehicle], float]:
        best_gv    = None
        best_score = 0.0

        for gv in self._registry:
            # Don't match to the same local track
            if track_id in gv.local_tracks.get(camera_id, []):
                continue
            if gv.appearance is None:
                continue
            score = compute_appearance_similarity(appearance, gv.appearance)
            if score > best_score:
                best_score = score
                best_gv    = gv

        return best_gv, best_score

    def _create(
        self,
        plate:         Optional[str],
        is_motorcycle: bool,
        appearance:    Optional[VehicleAppearance],
    ) -> GlobalVehicle:
        self._id_counter += 1
        gv = GlobalVehicle(
            global_id     = self._id_counter,
            plate         = plate,
            is_motorcycle = is_motorcycle,
            appearance    = appearance,
        )
        self._registry.append(gv)
        return gv

    # ── Registry API ───────────────────────────────────────────────

    def get_vehicle(self, global_id: int) -> Optional[GlobalVehicle]:
        for gv in self._registry:
            if gv.global_id == global_id:
                return gv
        return None

    def get_all_vehicles(self) -> list[GlobalVehicle]:
        return self._registry

    def get_motorcycles(self) -> list[GlobalVehicle]:
        return [gv for gv in self._registry if gv.is_motorcycle]

    def get_multi_camera_vehicles(self) -> list[GlobalVehicle]:
        """Vehicles seen on more than one camera — key for escape tracking."""
        return [gv for gv in self._registry if gv.camera_count > 1]

    def summary(self) -> dict:
        return {
            "total_vehicles":        len(self._registry),
            "motorcycles":           len(self.get_motorcycles()),
            "multi_camera_vehicles": len(self.get_multi_camera_vehicles()),
        }
