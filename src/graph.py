"""
graph.py — Spatiotemporal Movement Graph
=========================================
Every detected sighting of a vehicle becomes a node.
Edges represent movement between camera locations.

Node  → camera, timestamp, confidence, plate, embedding attributes
Edge  → time delta, haversine distance, is_inferred flag

Dev mode: GRAPH_GAP_FILL_ENABLED = False (no synthetic nodes).
          Edges are only drawn when gap < GRAPH_MAX_EDGE_GAP_SEC.

Schema (same premises, 6 cameras):
    cctv_01 ──→ cctv_03 ──→ cctv_05 …
    (Parking A Entry) → (Parking B North) → (Corridor East) …
"""

import math
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger("Graph")


# ══════════════════════════════════════════════════════════════════
# GEO HELPER
# ══════════════════════════════════════════════════════════════════

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class STNode:
    node_id:         int
    global_vehicle_id: int
    camera_id:       str
    location:        str
    timestamp:       datetime
    confidence:      float = 1.0
    plate:           Optional[str] = None
    is_gap_fill:     bool  = False

    # Embedding attributes (lightweight — no raw vectors stored here)
    vehicle_color:   str   = "unknown"
    helmet_present:  bool  = False
    helmet_color:    str   = "unknown"
    clothing_color:  str   = "unknown"
    vehicle_type:    str   = "unknown"
    plate_chars:     str   = ""


@dataclass
class STEdge:
    from_node:      int
    to_node:        int
    from_camera:    str
    to_camera:      str
    time_delta_sec: float
    distance_m:     float       # haversine metres between cameras
    is_inferred:    bool = False


# ══════════════════════════════════════════════════════════════════
# SPATIOTEMPORAL GRAPH
# ══════════════════════════════════════════════════════════════════

class SpatiotemporalGraph:
    """
    Builds a movement timeline graph for all vehicles across cameras.

    Usage:
        graph = SpatiotemporalGraph()
        graph.add_sighting(gv_id, camera_id, ts, conf, plate, embedding)
        graph.get_trajectory(gv_id)     → ordered list of nodes
        graph.get_edges_for_vehicle(gv_id)
        graph.to_dict()                 → full JSON-serialisable export
    """

    def __init__(self):
        self._nodes:   list[STNode] = []
        self._edges:   list[STEdge] = []
        self._counter: int = 0

        # last node id per vehicle — for fast edge connection
        self._last_node: dict[int, int] = {}   # gv_id → node_id

    # ── Main add ──────────────────────────────────────────────────

    def add_sighting(
        self,
        global_vehicle_id: int,
        camera_id:         str,
        timestamp:         datetime,
        confidence:        float = 1.0,
        plate:             Optional[str] = None,
        embedding=None,         # VehicleEmbedding | None
    ) -> STNode:
        """
        Record one camera sighting as a node and auto-draw edge from previous
        node for the same vehicle (if gap < GRAPH_MAX_EDGE_GAP_SEC).
        """
        self._counter += 1
        location = config.CAMERA_LOCATIONS.get(camera_id, camera_id)

        # Unpack embedding attributes
        v_color  = "unknown"
        h_present= False
        h_color  = "unknown"
        c_color  = "unknown"
        v_type   = "unknown"
        p_chars  = ""
        if embedding is not None:
            v_color   = embedding.dominant_color
            h_present = embedding.helmet_present
            h_color   = embedding.helmet_color
            c_color   = embedding.clothing_color
            v_type    = embedding.vehicle_type
            p_chars   = embedding.plate_chars

        node = STNode(
            node_id           = self._counter,
            global_vehicle_id = global_vehicle_id,
            camera_id         = camera_id,
            location          = location,
            timestamp         = timestamp,
            confidence        = confidence,
            plate             = plate,
            vehicle_color     = v_color,
            helmet_present    = h_present,
            helmet_color      = h_color,
            clothing_color    = c_color,
            vehicle_type      = v_type,
            plate_chars       = p_chars,
        )
        self._nodes.append(node)

        # Connect to previous node for this vehicle
        prev_id = self._last_node.get(global_vehicle_id)
        if prev_id is not None:
            prev_node = self._get_node(prev_id)
            if prev_node is not None:
                gap = (timestamp - prev_node.timestamp).total_seconds()
                if gap <= config.GRAPH_MAX_EDGE_GAP_SEC:
                    self._add_edge(prev_node, node, inferred=False)
                elif config.GRAPH_GAP_FILL_ENABLED:
                    # Production: insert synthetic midpoint node
                    fill = self._gap_fill_node(prev_node, node, global_vehicle_id)
                    self._add_edge(prev_node, fill, inferred=False)
                    self._add_edge(fill, node, inferred=True)
                else:
                    logger.debug(
                        f"[Graph] gv#{global_vehicle_id}: gap {gap:.0f}s "
                        f"exceeds limit — edge skipped."
                    )

        self._last_node[global_vehicle_id] = self._counter
        return node

    # ── Queries ────────────────────────────────────────────────────

    def get_trajectory(self, global_vehicle_id: int) -> list[dict]:
        """Ordered sightings for one vehicle (for dashboard timeline)."""
        nodes = sorted(
            [n for n in self._nodes if n.global_vehicle_id == global_vehicle_id],
            key=lambda n: n.timestamp,
        )
        return [self._node_to_dict(n, seq=i+1) for i, n in enumerate(nodes)]

    def get_edges_for_vehicle(self, global_vehicle_id: int) -> list[dict]:
        vnode_ids = {n.node_id for n in self._nodes if n.global_vehicle_id == global_vehicle_id}
        return [
            {
                "from_node":      e.from_node,
                "to_node":        e.to_node,
                "from_camera":    e.from_camera,
                "to_camera":      e.to_camera,
                "time_delta_sec": round(e.time_delta_sec, 1),
                "distance_m":     round(e.distance_m, 1),
                "is_inferred":    e.is_inferred,
            }
            for e in self._edges
            if e.from_node in vnode_ids or e.to_node in vnode_ids
        ]

    def get_all_vehicles_summary(self) -> list[dict]:
        """Per-vehicle summary: latest node attributes + camera count."""
        from collections import defaultdict
        by_vehicle: dict[int, list[STNode]] = defaultdict(list)
        for n in self._nodes:
            by_vehicle[n.global_vehicle_id].append(n)

        summary = []
        for gv_id, nodes in by_vehicle.items():
            latest = max(nodes, key=lambda n: n.timestamp)
            cameras = list({n.camera_id for n in nodes})
            summary.append({
                "global_vehicle_id": gv_id,
                "camera_count":      len(cameras),
                "cameras_seen":      cameras,
                "sighting_count":    len(nodes),
                "first_seen":        min(n.timestamp for n in nodes).isoformat(),
                "last_seen":         latest.timestamp.isoformat(),
                "last_camera":       latest.camera_id,
                "last_location":     latest.location,
                "vehicle_color":     latest.vehicle_color,
                "helmet_present":    latest.helmet_present,
                "helmet_color":      latest.helmet_color,
                "clothing_color":    latest.clothing_color,
                "vehicle_type":      latest.vehicle_type,
                "plate":             latest.plate,
            })
        return summary

    def to_dict(self) -> dict:
        """Full graph export for /api/graph endpoint."""
        return {
            "nodes": [self._node_to_dict(n) for n in self._nodes],
            "edges": [
                {
                    "from":           e.from_node,
                    "to":             e.to_node,
                    "from_camera":    e.from_camera,
                    "to_camera":      e.to_camera,
                    "time_delta_sec": round(e.time_delta_sec, 1),
                    "distance_m":     round(e.distance_m, 1),
                    "is_inferred":    e.is_inferred,
                }
                for e in self._edges
            ],
        }

    # ── Internal helpers ───────────────────────────────────────────

    def _add_edge(self, from_n: STNode, to_n: STNode, inferred: bool) -> None:
        delta = (to_n.timestamp - from_n.timestamp).total_seconds()
        c1 = config.CAMERA_COORDS.get(from_n.camera_id, (0.0, 0.0))
        c2 = config.CAMERA_COORDS.get(to_n.camera_id, (0.0, 0.0))
        dist_m = _haversine_km(c1[0], c1[1], c2[0], c2[1]) * 1000.0

        self._edges.append(STEdge(
            from_node      = from_n.node_id,
            to_node        = to_n.node_id,
            from_camera    = from_n.camera_id,
            to_camera      = to_n.camera_id,
            time_delta_sec = max(0.0, delta),
            distance_m     = dist_m,
            is_inferred    = inferred,
        ))

    def _gap_fill_node(self, n1: STNode, n2: STNode, gv_id: int) -> STNode:
        """Production-only: synthetic midpoint node between two sightings."""
        mid_ts = n1.timestamp + (n2.timestamp - n1.timestamp) / 2
        self._counter += 1
        c1 = config.CAMERA_COORDS.get(n1.camera_id, (0.0, 0.0))
        c2 = config.CAMERA_COORDS.get(n2.camera_id, (0.0, 0.0))
        mid_lat = (c1[0] + c2[0]) / 2
        mid_lon = (c1[1] + c2[1]) / 2

        node = STNode(
            node_id           = self._counter,
            global_vehicle_id = gv_id,
            camera_id         = "inferred",
            location          = f"Inferred ({mid_lat:.4f},{mid_lon:.4f})",
            timestamp         = mid_ts,
            confidence        = config.GAP_FILL_CONFIDENCE,
            is_gap_fill       = True,
        )
        self._nodes.append(node)
        return node

    def _get_node(self, node_id: int) -> Optional[STNode]:
        for n in self._nodes:
            if n.node_id == node_id:
                return n
        return None

    @staticmethod
    def _node_to_dict(n: STNode, seq: int = 0) -> dict:
        d = {
            "node_id":          n.node_id,
            "global_vehicle_id":n.global_vehicle_id,
            "camera_id":        n.camera_id,
            "location":         n.location,
            "timestamp":        n.timestamp.isoformat(),
            "confidence":       n.confidence,
            "plate":            n.plate,
            "is_gap_fill":      n.is_gap_fill,
            "vehicle_color":    n.vehicle_color,
            "helmet_present":   n.helmet_present,
            "helmet_color":     n.helmet_color,
            "clothing_color":   n.clothing_color,
            "vehicle_type":     n.vehicle_type,
            "plate_chars":      n.plate_chars,
        }
        if seq:
            d["seq"] = seq
        return d