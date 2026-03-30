"""
stream_manager.py — Per-Camera RTSP Stream Manager
====================================================
Manages 6 independent RTSP stream threads.
Each camera has:
  - A background frame-reader thread (non-blocking)
  - A frame queue for the pipeline to consume
  - start() / stop() controls
  - Auto-reconnect on drop

Architecture:
    [RTSP Camera] → [ReaderThread] → [FrameQueue] → [Pipeline]
"""

import cv2
import time
import queue
import logging
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger("StreamManager")


class StreamState(Enum):
    IDLE        = auto()
    CONNECTING  = auto()
    STREAMING   = auto()
    RECONNECTING= auto()
    STOPPED     = auto()
    ERROR       = auto()


@dataclass
class CameraStats:
    camera_id:    str
    state:        StreamState = StreamState.IDLE
    fps_actual:   float       = 0.0
    frames_read:  int         = 0
    dropped:      int         = 0
    errors:       int         = 0
    connected_at: Optional[datetime] = None
    last_frame_at:Optional[datetime] = None
    reconnects:   int         = 0


class CameraStream:
    """
    Single RTSP camera stream with a background reader thread.

    Usage:
        cam = CameraStream("cctv_01", "rtsp://...")
        cam.start()
        frame = cam.get_frame()  # None if no new frame
        cam.stop()
    """

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id  = camera_id
        self.rtsp_url   = rtsp_url
        self.location   = config.CAMERA_LOCATIONS.get(camera_id, camera_id)

        self._queue: queue.Queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = CameraStats(camera_id=camera_id)

    # ── Public API ──────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning(f"[{self.camera_id}] Already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"reader-{self.camera_id}",
            daemon=True
        )
        self.stats.state = StreamState.CONNECTING
        self._thread.start()
        logger.info(f"[{self.camera_id}] Started → {self.rtsp_url}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.stats.state = StreamState.STOPPED
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.info(f"[{self.camera_id}] Stopped.")

    def get_frame(self):
        """
        Non-blocking frame retrieval.
        Returns (frame_ndarray, timestamp) or (None, None).
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None, None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def state(self) -> StreamState:
        return self.stats.state

    # ── Internal reader loop ────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Background thread: connects, reads frames, handles reconnects."""
        fps_limit = config.STREAM_FPS_LIMIT
        min_delay = 1.0 / fps_limit if fps_limit > 0 else 0

        while not self._stop_event.is_set():
            cap = self._open_capture()
            if cap is None:
                self._wait_reconnect()
                continue

            self.stats.state        = StreamState.STREAMING
            self.stats.connected_at = datetime.now()
            self.stats.reconnects  += 1 if self.stats.reconnects > 0 else 0

            t_last = time.monotonic()
            fps_counter = 0
            fps_timer   = time.monotonic()

            logger.info(f"[{self.camera_id}] Stream connected.")

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                now = time.monotonic()

                if not ret:
                    logger.warning(f"[{self.camera_id}] Frame read failed — reconnecting.")
                    self.stats.errors += 1
                    break

                # FPS limiter
                elapsed = now - t_last
                if min_delay > 0 and elapsed < min_delay:
                    time.sleep(min_delay - elapsed)
                t_last = time.monotonic()

                self.stats.frames_read  += 1
                self.stats.last_frame_at = datetime.now()

                # FPS estimation
                fps_counter += 1
                if now - fps_timer >= 2.0:
                    self.stats.fps_actual = round(fps_counter / (now - fps_timer), 1)
                    fps_counter = 0
                    fps_timer   = now

                # Push to queue (drop oldest if full to maintain low latency)
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self.stats.dropped += 1
                    except queue.Empty:
                        pass

                try:
                    self._queue.put_nowait((frame, datetime.now()))
                except queue.Full:
                    pass

            cap.release()
            if not self._stop_event.is_set():
                self.stats.state = StreamState.RECONNECTING
                self._wait_reconnect()

        self.stats.state = StreamState.STOPPED

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open RTSP with TCP transport and optimal buffering."""
        import os
        # Force TCP at ffmpeg level — prevents H.264 decode errors over UDP
        if config.RTSP_TRANSPORT == "tcp":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)   # 10s connect timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)   # 10s read timeout

        if not cap.isOpened():
            logger.error(f"[{self.camera_id}] Cannot open stream: {self.rtsp_url}")
            self.stats.state  = StreamState.ERROR
            self.stats.errors += 1
            cap.release()
            return None
        return cap

    def _wait_reconnect(self) -> None:
        delay = config.RTSP_RECONNECT_MS / 1000
        logger.info(f"[{self.camera_id}] Reconnecting in {delay:.1f}s...")
        self.stats.reconnects += 1
        self._stop_event.wait(timeout=delay)


# ══════════════════════════════════════════════════════════════════
# STREAM MANAGER — Controls all 6 cameras
# ══════════════════════════════════════════════════════════════════

class StreamManager:
    """
    Manages all cameras defined in config.RTSP_CAMERAS.

    Usage:
        mgr = StreamManager()
        mgr.start_camera("cctv_01")
        mgr.start_all()
        frame, ts = mgr.get_frame("cctv_01")
        mgr.stop_camera("cctv_03")
        mgr.stop_all()
    """

    def __init__(self):
        self.cameras: dict[str, CameraStream] = {
            cam_id: CameraStream(cam_id, url)
            for cam_id, url in config.RTSP_CAMERAS.items()
        }
        logger.info(f"StreamManager initialised with {len(self.cameras)} cameras.")

    # ── Per-camera controls ─────────────────────────────────────────

    def start_camera(self, camera_id: str) -> bool:
        if camera_id not in self.cameras:
            logger.error(f"Unknown camera: {camera_id}")
            return False
        self.cameras[camera_id].start()
        return True

    def stop_camera(self, camera_id: str) -> bool:
        if camera_id not in self.cameras:
            logger.error(f"Unknown camera: {camera_id}")
            return False
        self.cameras[camera_id].stop()
        return True

    def restart_camera(self, camera_id: str) -> bool:
        self.stop_camera(camera_id)
        time.sleep(1)
        return self.start_camera(camera_id)

    # ── Bulk controls ───────────────────────────────────────────────

    def start_all(self) -> None:
        for cam_id in self.cameras:
            self.cameras[cam_id].start()
            time.sleep(0.3)  # stagger startup to avoid network spike

    def stop_all(self) -> None:
        for cam in self.cameras.values():
            cam.stop()

    # ── Frame access ────────────────────────────────────────────────

    def get_frame(self, camera_id: str):
        """Returns (frame, timestamp) or (None, None)."""
        if camera_id not in self.cameras:
            return None, None
        return self.cameras[camera_id].get_frame()

    def get_all_frames(self) -> dict:
        """Returns {camera_id: (frame, timestamp)} for all active cameras."""
        result = {}
        for cam_id, cam in self.cameras.items():
            if cam.state == StreamState.STREAMING:
                frame, ts = cam.get_frame()
                if frame is not None:
                    result[cam_id] = (frame, ts)
        return result

    # ── Status ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Returns serialisable status dict for all cameras."""
        status = {}
        for cam_id, cam in self.cameras.items():
            s = cam.stats
            status[cam_id] = {
                "camera_id":    cam_id,
                "location":     cam.location,
                "state":        s.state.name,
                "fps":          s.fps_actual,
                "frames_read":  s.frames_read,
                "dropped":      s.dropped,
                "errors":       s.errors,
                "reconnects":   s.reconnects,
                "connected_at": s.connected_at.isoformat() if s.connected_at else None,
                "last_frame_at":s.last_frame_at.isoformat() if s.last_frame_at else None,
            }
        return status

    def get_active_camera_ids(self) -> list[str]:
        return [
            cam_id for cam_id, cam in self.cameras.items()
            if cam.state == StreamState.STREAMING
        ]


# ── Singleton instance ─────────────────────────────────────────────
_manager: Optional[StreamManager] = None

def get_stream_manager() -> StreamManager:
    global _manager
    if _manager is None:
        _manager = StreamManager()
    return _manager