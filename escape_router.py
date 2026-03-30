"""
escape_router.py — Escape Route Prediction + Spatiotemporal Graph
=================================================================
Given a confirmed fleeing vehicle:

  1. EscapeRouter      — predicts top-N escape routes from crash location
                         using camera GPS coords and city topology (or fallback
                         heuristic if road-graph not present)

  2. CameraOrchestrator— dynamically prioritises cameras along predicted routes

  3. SpatiotemporalGraph— builds movement timeline with gap-filling for blind spots
"""

import json
import math
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import config
from reid import GlobalVehicle, CameraSighting

logger = logging.getLogger("EscapeRouter")


# ══════════════════════════════════════════════════════════════════
# GEO HELPERS
# ══════════════════════════════════════════════════════════════════

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two GPS points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point1 to point2 (degrees)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ══════════════════════════════════════════════════════════════════
# ESCAPE ROUTE
# ══════════════════════════════════════════════════════════════════

@dataclass
class RouteSegment:
    from_camera:   str
    to_camera:     str
    distance_km:   float
    bearing_deg:   float
    travel_time_min: float       # estimated at avg motorcycle speed
    cameras_on_route: list[str]  # intermediate cameras to scan


@dataclass
class EscapeRoute:
    route_id:   int
    segments:   list[RouteSegment] = field(default_factory=list)
    probability: float = 0.0
    total_km:   float  = 0.0
    description: str   = ""


class EscapeRouter:
    """
    Predicts most likely escape routes for a fleeing motorcycle.
    Uses camera GPS positions to build a proximity graph.
    Falls back to heuristic if no OSM road graph is available.
    """

    AVG_MOTORCYCLE_SPEED_KMH = 60.0

    def __init__(self):
        self._road_graph = self._try_load_road_graph()
        self._camera_coords = config.CAMERA_COORDS
        logger.info(
            f"EscapeRouter ready  |  "
            f"{'OSMnx graph loaded' if self._road_graph else 'Heuristic mode (no road graph)'}"
        )

    def _try_load_road_graph(self) -> Optional[dict]:
        try:
            p = config.CITY_GRAPH_PATH
            with open(p) as f:
                graph = json.load(f)
            logger.info(f"City road graph loaded from {p}")
            return graph
        except Exception:
            return None

    def predict_routes(
        self,
        incident_camera: str,
        vehicle: GlobalVehicle,
        n_routes: int = None,
    ) -> list[EscapeRoute]:
        """
        Returns top-N predicted escape routes from the incident camera.
        """
        n_routes = n_routes or config.ESCAPE_MAX_ROUTES

        if incident_camera not in self._camera_coords:
            logger.warning(f"No GPS for camera {incident_camera}")
            return []

        origin = self._camera_coords[incident_camera]

        # Compute distances from incident point to all other cameras
        candidates = []
        for cam_id, coords in self._camera_coords.items():
            if cam_id == incident_camera:
                continue
            dist   = haversine(origin[0], origin[1], coords[0], coords[1])
            brng   = bearing(origin[0], origin[1], coords[0], coords[1])
            travel = (dist / self.AVG_MOTORCYCLE_SPEED_KMH) * 60  # minutes
            candidates.append((cam_id, dist, brng, travel))

        # Sort by distance (nearest exits first)
        candidates.sort(key=lambda x: x[1])

        # If we have sighting history, bias toward observed direction
        direction_bias = self._compute_direction_bias(vehicle)

        routes = []
        for i, (cam_id, dist, brng, travel) in enumerate(candidates[:n_routes]):
            # Adjust probability by direction alignment
            if direction_bias is not None:
                angle_diff = abs(brng - direction_bias)
                angle_diff = min(angle_diff, 360 - angle_diff)
                prob = max(0.1, 1.0 - (angle_diff / 180.0))
            else:
                prob = 1.0 / (i + 1)

            segment = RouteSegment(
                from_camera      = incident_camera,
                to_camera        = cam_id,
                distance_km      = round(dist, 2),
                bearing_deg      = round(brng, 1),
                travel_time_min  = round(travel, 1),
                cameras_on_route = self._cameras_on_path(incident_camera, cam_id),
            )

            route = EscapeRoute(
                route_id    = i + 1,
                segments    = [segment],
                probability = round(prob, 3),
                total_km    = round(dist, 2),
                description = (
                    f"Route {i+1}: {config.CAMERA_LOCATIONS.get(incident_camera, incident_camera)} "
                    f"→ {config.CAMERA_LOCATIONS.get(cam_id, cam_id)}  "
                    f"({dist:.1f}km, ~{travel:.0f}min)"
                ),
            )
            routes.append(route)

        # Normalise probabilities
        total_prob = sum(r.probability for r in routes)
        if total_prob > 0:
            for r in routes:
                r.probability = round(r.probability / total_prob, 3)

        routes.sort(key=lambda x: -x.probability)
        return routes

    def _compute_direction_bias(self, vehicle: GlobalVehicle) -> Optional[float]:
        """Estimate direction of travel from multi-camera sightings."""
        sightings = sorted(vehicle.sightings, key=lambda s: s.timestamp)
        if len(sightings) < 2:
            return None

        last = sightings[-1]
        prev = sightings[-2]

        if last.camera_id not in self._camera_coords or prev.camera_id not in self._camera_coords:
            return None

        c1 = self._camera_coords[prev.camera_id]
        c2 = self._camera_coords[last.camera_id]
        return bearing(c1[0], c1[1], c2[0], c2[1])

    def _cameras_on_path(self, from_cam: str, to_cam: str) -> list[str]:
        """Return cameras geographically between two endpoints."""
        if from_cam not in self._camera_coords or to_cam not in self._camera_coords:
            return []

        c1 = self._camera_coords[from_cam]
        c2 = self._camera_coords[to_cam]
        mid_lat = (c1[0] + c2[0]) / 2
        mid_lon = (c1[1] + c2[1]) / 2
        threshold_km = haversine(c1[0], c1[1], c2[0], c2[1]) / 2

        intermediate = []
        for cam_id, coords in self._camera_coords.items():
            if cam_id in (from_cam, to_cam):
                continue
            d = haversine(mid_lat, mid_lon, coords[0], coords[1])
            if d <= threshold_km:
                intermediate.append(cam_id)
        return intermediate


# ══════════════════════════════════════════════════════════════════
# CAMERA ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

class CameraOrchestrator:
    """
    Dynamically assigns priority scores to cameras based on escape routes.
    High-priority cameras should receive more processing budget.
    """

    def __init__(self, all_camera_ids: list[str]):
        self._base_priority  = {cam: 1 for cam in all_camera_ids}
        self._active_boosts: dict[str, float] = {}  # cam_id -> boost expiry epoch

    def apply_escape_routes(self, routes: list[EscapeRoute]) -> None:
        """Boost cameras on predicted escape routes."""
        import time
        expiry = time.time() + config.ESCAPE_HORIZON_MIN * 60

        for route in routes:
            boost = int(config.FLEE_CAMERA_PRIORITY_BOOST * route.probability * 10)
            for seg in route.segments:
                self._active_boosts[seg.to_camera]  = expiry
                for cam in seg.cameras_on_route:
                    self._active_boosts[cam] = expiry
        logger.info(f"Camera boosts applied to: {list(self._active_boosts.keys())}")

    def get_priority(self, camera_id: str) -> int:
        import time
        base  = self._base_priority.get(camera_id, 1)
        boost = 0
        exp   = self._active_boosts.get(camera_id)
        if exp and time.time() < exp:
            boost = config.FLEE_CAMERA_PRIORITY_BOOST
        return base + boost

    def get_all_priorities(self) -> dict[str, int]:
        return {cam: self.get_priority(cam) for cam in self._base_priority}


# ══════════════════════════════════════════════════════════════════
# SPATIOTEMPORAL GRAPH
# ══════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    node_id:    int
    camera_id:  str
    location:   str
    timestamp:  datetime
    track_id:   int
    global_id:  int
    plate:      Optional[str]
    is_gap_fill:bool   = False      # True = probabilistically inferred
    confidence: float  = 1.0


@dataclass
class GraphEdge:
    from_node:  int
    to_node:    int
    travel_sec: float
    distance_km:float
    is_inferred:bool  = False


class SpatiotemporalGraph:
    """
    Builds a movement timeline graph for a vehicle across all cameras.
    Automatically gap-fills blind segments with probabilistic nodes.
    """

    def __init__(self):
        self._nodes: list[GraphNode] = []
        self._edges: list[GraphEdge] = []
        self._node_counter = 0

    def add_sighting(self, sighting: CameraSighting, global_id: int) -> GraphNode:
        self._node_counter += 1
        node = GraphNode(
            node_id    = self._node_counter,
            camera_id  = sighting.camera_id,
            location   = sighting.location or sighting.camera_id,
            timestamp  = sighting.timestamp,
            track_id   = sighting.track_id,
            global_id  = global_id,
            plate      = sighting.plate,
        )
        self._nodes.append(node)

        # Auto-connect to previous node for same vehicle
        prev_nodes = [
            n for n in self._nodes[:-1]
            if n.global_id == global_id
        ]
        if prev_nodes:
            prev = max(prev_nodes, key=lambda n: n.timestamp)
            gap_sec = (sighting.timestamp - prev.timestamp).total_seconds()

            if gap_sec > config.GAP_FILL_MAX_SEC:
                # Gap too large — insert a synthetic inferred node
                inferred = self._fill_gap(prev, node, global_id)
                self._add_edge(prev.node_id, inferred.node_id, inferred=False)
                self._add_edge(inferred.node_id, node.node_id, inferred=True)
            else:
                self._add_edge(prev.node_id, node.node_id, inferred=False)

        return node

    def _fill_gap(self, n1: GraphNode, n2: GraphNode, global_id: int) -> GraphNode:
        """Insert a gap-fill node halfway between two sightings."""
        mid_time = n1.timestamp + (n2.timestamp - n1.timestamp) / 2
        self._node_counter += 1

        # Infer location from midpoint of GPS coords
        c1 = config.CAMERA_COORDS.get(n1.camera_id, (0, 0))
        c2 = config.CAMERA_COORDS.get(n2.camera_id, (0, 0))
        mid_lat = (c1[0] + c2[0]) / 2
        mid_lon = (c1[1] + c2[1]) / 2

        node = GraphNode(
            node_id     = self._node_counter,
            camera_id   = "inferred",
            location    = f"Inferred ({mid_lat:.4f},{mid_lon:.4f})",
            timestamp   = mid_time,
            track_id    = -1,
            global_id   = global_id,
            plate       = None,
            is_gap_fill = True,
            confidence  = config.GAP_FILL_CONFIDENCE,
        )
        self._nodes.append(node)
        logger.debug(f"[Graph] Gap-fill node inserted between {n1.camera_id} → {n2.camera_id}")
        return node

    def _add_edge(self, from_id: int, to_id: int, inferred: bool = False) -> None:
        n1 = next((n for n in self._nodes if n.node_id == from_id), None)
        n2 = next((n for n in self._nodes if n.node_id == to_id),   None)
        if n1 is None or n2 is None:
            return

        travel_sec  = (n2.timestamp - n1.timestamp).total_seconds()
        c1 = config.CAMERA_COORDS.get(n1.camera_id, (0, 0))
        c2 = config.CAMERA_COORDS.get(n2.camera_id, (0, 0))
        dist_km = haversine(c1[0], c1[1], c2[0], c2[1]) if c1 != (0,0) else 0.0

        self._edges.append(GraphEdge(
            from_node   = from_id,
            to_node     = to_id,
            travel_sec  = travel_sec,
            distance_km = round(dist_km, 2),
            is_inferred = inferred,
        ))

    # ── Export ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id":         n.node_id,
                    "camera_id":  n.camera_id,
                    "location":   n.location,
                    "timestamp":  n.timestamp.isoformat(),
                    "global_id":  n.global_id,
                    "plate":      n.plate,
                    "gap_fill":   n.is_gap_fill,
                    "confidence": n.confidence,
                }
                for n in self._nodes
            ],
            "edges": [
                {
                    "from":       e.from_node,
                    "to":         e.to_node,
                    "travel_sec": e.travel_sec,
                    "km":         e.distance_km,
                    "inferred":   e.is_inferred,
                }
                for e in self._edges
            ],
        }

    def get_timeline(self, global_id: int) -> list[dict]:
        """Ordered timeline for one vehicle."""
        nodes = sorted(
            [n for n in self._nodes if n.global_id == global_id],
            key=lambda n: n.timestamp
        )
        return [
            {
                "seq":       i + 1,
                "location":  n.location,
                "camera_id": n.camera_id,
                "timestamp": n.timestamp.isoformat(),
                "plate":     n.plate,
                "gap_fill":  n.is_gap_fill,
                "confidence":n.confidence,
            }
            for i, n in enumerate(nodes)
        ]
