"""
detector.py — Vehicle Detection + ByteTrack + Anomaly Engine
=============================================================
Pipeline per frame:
    [Frame] → [YOLO Plate Detection] → [ByteTrack] → [Anomaly Check] → [Events]

Outputs per detection:
    - Track ID (persistent across frames)
    - Vehicle class (motorcycle / car / etc.)
    - Bounding box + confidence
    - Speed estimate (pixels/frame)
    - Anomaly flags (crash, fleeing, wrong-way)
"""

import cv2
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import defaultdict, deque

from ultralytics import YOLO

import config

logger = logging.getLogger("Detector")


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    track_id:   int
    class_id:   int
    class_name: str
    bbox:       tuple          # (x1, y1, x2, y2)
    conf:       float
    camera_id:  str
    timestamp:  datetime
    frame_idx:  int
    is_motorcycle: bool = False
    center:     tuple = (0, 0)

    # Motion metrics
    speed_px:   float = 0.0   # pixels/frame
    direction:  float = 0.0   # degrees (0=right, 90=up)

    # Anomaly flags
    is_fleeing: bool  = False
    is_crash:   bool  = False
    wrong_way:  bool  = False
    anomaly_score: float = 0.0
    
    # Padded vehicle crop for LPR + ReID
    crop: Optional[np.ndarray] = None

@dataclass
class TrackHistory:
    """Sliding window history per track ID."""
    track_id:   int
    centers:    deque = field(default_factory=lambda: deque(maxlen=30))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=30))
    speeds:     deque = field(default_factory=lambda: deque(maxlen=15))
    directions: deque = field(default_factory=lambda: deque(maxlen=15))
    camera_ids: deque = field(default_factory=lambda: deque(maxlen=30))


@dataclass
class IncidentEvent:
    event_type:  str               # "crash" | "flee" | "wrong_way" | "anomaly"
    camera_id:   str
    track_id:    int
    timestamp:   datetime
    bbox:        tuple
    frame:       Optional[np.ndarray] = None
    description: str = ""
    confidence:  float = 0.0


# ══════════════════════════════════════════════════════════════════
# VEHICLE DETECTOR
# ══════════════════════════════════════════════════════════════════

class VehicleDetector:
    """
    Wraps YOLOv8 + ByteTrack for multi-camera plate tracking.
    Uses your single plate detector model (best.pt).
    Each camera gets its own frame counter and track history
    so tracks don't collide across cameras.
    """

    def __init__(self):
        if not Path(config.PLATE_WEIGHTS).exists():
            logger.warning(
                f"Plate weights not found: {config.PLATE_WEIGHTS}. "
                "Put your best.pt in the weights/ folder."
            )
            self.model = None
        else:
            logger.info(f"Loading plate detector: {config.PLATE_WEIGHTS}")
            self.model = YOLO(config.PLATE_WEIGHTS)
            logger.info("Plate detector ready.")

        # Per-camera state
        self._frame_counters: dict[str, int]  = defaultdict(int)
        self._track_history:  dict[str, dict] = defaultdict(dict)  # camera_id -> {track_id: TrackHistory}
        self._area_speeds:    dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # Per-camera last results (for frame-skip persistence)
        self._last_detections: dict[str, list[Detection]] = defaultdict(list)

        # Incident event queue (consumed by pipeline)
        self._incident_events: list[IncidentEvent] = []
        self._last_incident_ts: dict[str, float] = defaultdict(float)  # camera_id -> epoch

    # ── Main entry point ───────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp: Optional[datetime] = None,
    ) -> tuple[list[Detection], np.ndarray]:
        """
        Process one frame from a camera.
        Returns:
            detections  — list of Detection objects this frame
            annotated   — frame with overlays drawn
        """
        if timestamp is None:
            timestamp = datetime.now()

        self._frame_counters[camera_id] += 1
        frame_idx = self._frame_counters[camera_id]

        # Frame-skip: re-use last detections on non-inference frames
        if frame_idx % config.SKIP_FRAMES != 0 or self.model is None:
            detections = self._last_detections[camera_id]
            annotated  = self._draw(frame.copy(), detections)
            return detections, annotated

        # ── YOLO + ByteTrack ──────────────────────────────────────
        results = self.model.track(
            frame,
            conf=config.YOLO_CONF,
            imgsz=config.YOLO_IMG_SIZE,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )[0]

        detections: list[Detection] = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes  = results.boxes.xyxy.cpu().numpy()
            ids    = results.boxes.id.cpu().numpy().astype(int)
            confs  = results.boxes.conf.cpu().numpy()
            clsids = results.boxes.cls.cpu().numpy().astype(int)

            for box, tid, conf, cid in zip(boxes, ids, confs, clsids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                det = Detection(
                    track_id    = tid,
                    class_id    = int(cid),
                    class_name  = "plate",
                    bbox        = (x1, y1, x2, y2),
                    conf        = round(float(conf), 3),
                    camera_id   = camera_id,
                    timestamp   = timestamp,
                    frame_idx   = frame_idx,
                    is_motorcycle = True,   # plate model tracks vehicles — treat all as targets
                    center      = (cx, cy),
                )
                # Extract padded crop (used by LPR + ReID in pipeline)
                pad  = config.CROP_PAD_PX
                fh, fw = frame.shape[:2]
                det.crop = frame[
                    max(0, y1 - pad) : min(fh, y2 + pad),
                    max(0, x1 - pad) : min(fw, x2 + pad),
                ].copy()

                # Compute speed + direction from history
                self._update_track_history(camera_id, tid, (cx, cy), timestamp, det)

                # Anomaly analysis
                self._analyse_anomaly(det, camera_id)

                detections.append(det)

        # Check for crash (high IoU between tracks)
        self._check_collision(detections, camera_id, frame, timestamp)

        self._last_detections[camera_id] = detections
        annotated = self._draw(frame.copy(), detections)
        return detections, annotated

    # ── Track history + motion ─────────────────────────────────────

    def _update_track_history(
        self, camera_id: str, tid: int,
        center: tuple, ts: datetime, det: Detection
    ) -> None:
        if tid not in self._track_history[camera_id]:
            self._track_history[camera_id][tid] = TrackHistory(track_id=tid)

        h = self._track_history[camera_id][tid]
        h.centers.append(center)
        h.timestamps.append(ts)
        h.camera_ids.append(camera_id)

        if len(h.centers) >= 2:
            dx = center[0] - h.centers[-2][0]
            dy = center[1] - h.centers[-2][1]
            speed   = float(np.hypot(dx, dy))
            direction = float(np.degrees(np.arctan2(-dy, dx)) % 360)

            h.speeds.append(speed)
            h.directions.append(direction)
            self._area_speeds[camera_id].append(speed)

            det.speed_px  = round(float(np.mean(h.speeds)), 2)
            det.direction = round(direction, 1)

    # ── Anomaly detection ──────────────────────────────────────────

    def _analyse_anomaly(self, det: Detection, camera_id: str) -> None:
        score = 0.0

        # Fleeing: vehicle much faster than local average
        if self._area_speeds[camera_id]:
            avg_speed = float(np.mean(self._area_speeds[camera_id]))
            if avg_speed > 0 and det.speed_px > avg_speed * config.FLEE_SPEED_MULTIPLIER:
                det.is_fleeing = True
                score += 0.4

        # Direction reversal (wrong-way / sudden U-turn)
        if config.ANOMALY_DIRECTION_FLIP:
            h = self._track_history[camera_id].get(det.track_id)
            if h and len(h.directions) >= 5:
                recent_dirs = list(h.directions)[-5:]
                diffs = [abs(recent_dirs[i] - recent_dirs[i-1]) for i in range(1, len(recent_dirs))]
                if any(d > 120 for d in diffs):
                    det.wrong_way = True
                    score += 0.35

        det.anomaly_score = min(round(score, 2), 1.0)

        # Emit fleeing event (with cooldown)
        if det.is_fleeing and det.is_motorcycle:
            now = time.time()
            if now - self._last_incident_ts[camera_id] > config.INCIDENT_COOLDOWN_SEC:
                self._last_incident_ts[camera_id] = now
                self._incident_events.append(IncidentEvent(
                    event_type  = "flee",
                    camera_id   = camera_id,
                    track_id    = det.track_id,
                    timestamp   = det.timestamp,
                    bbox        = det.bbox,
                    description = f"Motorcycle ID#{det.track_id} detected fleeing at {det.speed_px:.1f}px/fr",
                    confidence  = det.anomaly_score,
                ))

    def _check_collision(
        self, detections: list[Detection],
        camera_id: str, frame: np.ndarray, ts: datetime
    ) -> None:
        """Detect pairwise vehicle overlap → potential crash."""
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._iou(detections[i].bbox, detections[j].bbox)
                if iou > config.ANOMALY_OVERLAP_THRESH:
                    detections[i].is_crash = True
                    detections[j].is_crash = True
                    now = time.time()
                    if now - self._last_incident_ts[camera_id] > config.INCIDENT_COOLDOWN_SEC:
                        self._last_incident_ts[camera_id] = now
                        self._incident_events.append(IncidentEvent(
                            event_type  = "crash",
                            camera_id   = camera_id,
                            track_id    = detections[i].track_id,
                            timestamp   = ts,
                            bbox        = detections[i].bbox,
                            frame       = frame.copy(),
                            description = (
                                f"Collision detected between ID#{detections[i].track_id} "
                                f"and ID#{detections[j].track_id}  (IoU={iou:.2f})"
                            ),
                            confidence  = min(iou * 1.5, 1.0),
                        ))

    # ── Events API ─────────────────────────────────────────────────

    def pop_events(self) -> list[IncidentEvent]:
        """Drain and return all queued incident events."""
        events = self._incident_events.copy()
        self._incident_events.clear()
        return events

    # ── Drawing ────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Color logic
            if det.is_crash:
                color = (0, 0, 255)      # red — crash
            elif det.is_fleeing:
                color = (0, 165, 255)    # orange — fleeing
            elif det.wrong_way:
                color = (0, 0, 200)      # dark red — wrong way
            elif det.is_motorcycle:
                color = (255, 200, 0)    # cyan-ish — motorcycle
            else:
                color = (0, 200, 100)    # green — normal vehicle

            # Bounding box
            thickness = 3 if (det.is_crash or det.is_fleeing) else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label
            label = f"#{det.track_id} {det.class_name} {det.conf:.2f}"
            if det.is_fleeing:
                label += " ⚠ FLEE"
            if det.is_crash:
                label += " 🔴 CRASH"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # Speed indicator
            if det.speed_px > 0:
                spd_label = f"spd:{det.speed_px:.1f}"
                cv2.putText(frame, spd_label, (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        # Camera overlay
        cv2.putText(frame, f"SENTINEL | Active Tracks: {len(detections)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _iou(b1: tuple, b2: tuple) -> float:
        ax1, ay1, ax2, ay2 = b1
        bx1, by1, bx2, by2 = b2
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0: return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)