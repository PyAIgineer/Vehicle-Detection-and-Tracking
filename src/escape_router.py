"""
escape_router.py — Escape Route Prediction + Camera Orchestration
=================================================================
SpatiotemporalGraph has been moved to graph.py.

EscapeRouter    — predicts top-N routes from incident camera using
                  GPS coords of the 6 on-premises cameras.
                  In dev: all cameras are ~same site so distances
                  are small — useful for testing the pipeline logic.

CameraOrchestrator — priority scoring per camera based on route.
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger("EscapeRouter")


# ══════════════════════════════════════════════════════════════════
# GEO HELPERS
# ══════════════════════════════════════════════════════════════════

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x  = math.sin(dl) * math.cos(p2)
    y  = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class RouteSegment:
    from_camera:      str
    to_camera:        str
    distance_km:      float
    bearing_deg:      float
    travel_time_min:  float
    cameras_on_route: list[str] = field(default_factory=list)


@dataclass
class EscapeRoute:
    route_id:    int
    segments:    list[RouteSegment] = field(default_factory=list)
    probability: float = 0.0
    total_km:    float = 0.0
    description: str   = ""


# ══════════════════════════════════════════════════════════════════
# ESCAPE ROUTER
# ══════════════════════════════════════════════════════════════════

class EscapeRouter:
    """
    Predicts most likely next camera a vehicle will appear at,
    based on GPS positions and last known movement direction.
    """

    AVG_SPEED_KMH = 30.0   # conservative for parking lot / premises

    def __init__(self):
        self._coords = config.CAMERA_COORDS
        logger.info("EscapeRouter ready (GPS heuristic mode).")

    def predict_routes(
        self,
        incident_camera: str,
        vehicle,                       # GlobalVehicle
        n_routes: int = None,
    ) -> list[EscapeRoute]:
        n_routes = n_routes or config.ESCAPE_MAX_ROUTES

        if incident_camera not in self._coords:
            logger.warning(f"No GPS for camera {incident_camera}")
            return []

        origin = self._coords[incident_camera]
        direction_bias = self._direction_bias(vehicle)

        candidates = []
        for cam_id, coords in self._coords.items():
            if cam_id == incident_camera:
                continue
            dist  = haversine_km(origin[0], origin[1], coords[0], coords[1])
            brng  = bearing_deg(origin[0], origin[1], coords[0], coords[1])
            travel = (dist / self.AVG_SPEED_KMH) * 60
            candidates.append((cam_id, dist, brng, travel))

        candidates.sort(key=lambda x: x[1])

        routes = []
        for i, (cam_id, dist, brng, travel) in enumerate(candidates[:n_routes]):
            if direction_bias is not None:
                diff = abs(brng - direction_bias)
                diff = min(diff, 360 - diff)
                prob = max(0.1, 1.0 - diff / 180.0)
            else:
                prob = 1.0 / (i + 1)

            seg = RouteSegment(
                from_camera      = incident_camera,
                to_camera        = cam_id,
                distance_km      = round(dist, 4),
                bearing_deg      = round(brng, 1),
                travel_time_min  = round(travel, 2),
                cameras_on_route = self._intermediate(incident_camera, cam_id),
            )
            routes.append(EscapeRoute(
                route_id    = i + 1,
                segments    = [seg],
                probability = round(prob, 3),
                total_km    = round(dist, 4),
                description = (
                    f"Route {i+1}: "
                    f"{config.CAMERA_LOCATIONS.get(incident_camera, incident_camera)} → "
                    f"{config.CAMERA_LOCATIONS.get(cam_id, cam_id)} "
                    f"({dist*1000:.0f}m, ~{travel*60:.0f}s)"
                ),
            ))

        total = sum(r.probability for r in routes)
        if total > 0:
            for r in routes:
                r.probability = round(r.probability / total, 3)

        routes.sort(key=lambda x: -x.probability)
        return routes

    def _direction_bias(self, vehicle) -> Optional[float]:
        sightings = sorted(vehicle.sightings, key=lambda s: s.timestamp)
        if len(sightings) < 2:
            return None
        c_prev = self._coords.get(sightings[-2].camera_id)
        c_last = self._coords.get(sightings[-1].camera_id)
        if c_prev is None or c_last is None:
            return None
        return bearing_deg(c_prev[0], c_prev[1], c_last[0], c_last[1])

    def _intermediate(self, from_cam: str, to_cam: str) -> list[str]:
        c1 = self._coords.get(from_cam)
        c2 = self._coords.get(to_cam)
        if not c1 or not c2:
            return []
        mid_lat = (c1[0] + c2[0]) / 2
        mid_lon = (c1[1] + c2[1]) / 2
        thresh  = haversine_km(c1[0], c1[1], c2[0], c2[1]) / 2
        return [
            cam for cam, coords in self._coords.items()
            if cam not in (from_cam, to_cam)
            and haversine_km(mid_lat, mid_lon, coords[0], coords[1]) <= thresh
        ]


# ══════════════════════════════════════════════════════════════════
# CAMERA ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

class CameraOrchestrator:
    """Priority scoring for cameras based on active escape routes."""

    def __init__(self, all_camera_ids: list[str]):
        self._base    = {cam: 1 for cam in all_camera_ids}
        self._boosts: dict[str, float] = {}   # cam_id → expiry epoch

    def apply_escape_routes(self, routes: list[EscapeRoute]) -> None:
        expiry = time.time() + config.ESCAPE_HORIZON_MIN * 60
        for route in routes:
            for seg in route.segments:
                self._boosts[seg.to_camera] = expiry
                for cam in seg.cameras_on_route:
                    self._boosts[cam] = expiry
        logger.info(f"Camera boosts applied: {list(self._boosts.keys())}")

    def get_priority(self, camera_id: str) -> int:
        base  = self._base.get(camera_id, 1)
        exp   = self._boosts.get(camera_id)
        boost = config.FLEE_CAMERA_PRIORITY_BOOST if (exp and time.time() < exp) else 0
        return base + boost

    def get_all_priorities(self) -> dict[str, int]:
        return {cam: self.get_priority(cam) for cam in self._base}