"""
main.py — Sentinel Incident Intelligence System — FastAPI Server
================================================================
Endpoints:
  GET  /api/cameras              — list all cameras + status
  POST /api/cameras/{id}/start   — start stream
  POST /api/cameras/{id}/stop    — stop stream
  POST /api/cameras/start-all    — start all 6 cameras
  POST /api/cameras/stop-all     — stop all cameras
  GET  /api/incidents            — list all incidents
  GET  /api/incidents/{id}       — get incident report
  GET  /api/vehicles             — all global vehicles tracked
  GET  /api/vehicles/{id}        — specific vehicle with movement graph
  WS   /ws/camera/{id}           — JPEG frame stream for one camera
  WS   /ws/alerts                — realtime alert events

Run:
  pip install fastapi uvicorn python-dotenv
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import cv2
import time
import asyncio
import base64
import logging
import numpy as np
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# Suppress noisy uvicorn access log (HTTP GET/POST spam)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# Only show WARNING+ from these chatty libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logger = logging.getLogger("SentinelAPI")

import config
from stream_manager import StreamManager, StreamState
from detector       import VehicleDetector, IncidentEvent
from lpr_pipeline   import LPRPipeline
from reid           import ReIDEngine
from escape_router  import EscapeRouter, CameraOrchestrator, SpatiotemporalGraph
from incident_report import ReportGenerator, IncidentReport


# ══════════════════════════════════════════════════════════════════
# APP STATE
# ══════════════════════════════════════════════════════════════════

class AppState:
    def __init__(self):
        self.stream_mgr    = StreamManager()
        self.detector      = VehicleDetector()
        self.lpr           = LPRPipeline()
        self.reid_engine   = ReIDEngine()
        self.escape_router = EscapeRouter()
        self.orchestrator  = CameraOrchestrator(list(config.RTSP_CAMERAS.keys()))
        self.graph         = SpatiotemporalGraph()
        self.report_gen    = ReportGenerator()

        self.incidents:   list[IncidentReport] = []
        self.alert_queue: asyncio.Queue        = asyncio.Queue()

        # WS connection registries
        self.camera_ws:  dict[str, set[WebSocket]] = {c: set() for c in config.RTSP_CAMERAS}
        self.alert_ws:   set[WebSocket]             = set()


state: Optional[AppState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    state = AppState()
    logger.info("Sentinel system initialised.")
    asyncio.create_task(pipeline_loop())
    yield
    state.stream_mgr.stop_all()
    logger.info("Sentinel system shutdown.")


app = FastAPI(
    title="Sentinel Incident Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dashboard
# Auto-create frontend dir and mount only if it exists
Path("frontend").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ══════════════════════════════════════════════════════════════════
# PIPELINE LOOP
# ══════════════════════════════════════════════════════════════════

async def pipeline_loop():
    """
    Main async loop: reads frames from all active cameras,
    runs detection/LPR/ReID, pushes frames to WS clients,
    processes incident events.
    """
    ws_delay = 1.0 / config.WEBSOCKET_FPS

    while True:
        frame_start = time.monotonic()

        for cam_id in state.stream_mgr.get_active_camera_ids():
            frame, ts = state.stream_mgr.get_frame(cam_id)
            if frame is None:
                continue

            # ── Detection pipeline ──────────────────────────────
            detections, annotated = state.detector.process_frame(frame, cam_id, ts)

            # ── LPR: detector already found plate bbox, pass crop directly ──
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                # Expand crop slightly for better OCR context
                h, w = frame.shape[:2]
                pad = 6
                crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
                lpr_results = state.lpr.process_frame(crop, cam_id, det.track_id, ts)

                plate_str = None
                if lpr_results and lpr_results[0].valid:
                    plate_str = lpr_results[0].plate
                    annotated = state.lpr.draw_overlay(annotated, lpr_results)

                # ── ReID ────────────────────────────────────────
                gv = state.reid_engine.match_or_create(
                    camera_id     = cam_id,
                    track_id      = det.track_id,
                    crop          = crop,
                    plate         = plate_str,
                    is_motorcycle = det.is_motorcycle,
                    timestamp     = ts,
                    bbox          = det.bbox,
                )

                # ── Spatiotemporal graph ─────────────────────────
                sighting = gv.sightings[-1]
                state.graph.add_sighting(sighting, gv.global_id)

            # ── Push annotated frame to WS clients ───────────────
            if state.camera_ws.get(cam_id):
                jpg_bytes = _encode_frame(annotated)
                await _broadcast_frame(cam_id, jpg_bytes)

        # ── Process incident events ─────────────────────────────
        events = state.detector.pop_events()
        for event in events:
            await _handle_incident(event)

        # ── Maintain target FPS ─────────────────────────────────
        elapsed = time.monotonic() - frame_start
        await asyncio.sleep(max(0, ws_delay - elapsed))


async def _handle_incident(event: IncidentEvent):
    """Generate report and push alert for a detected incident event."""
    logger.warning(f"[INCIDENT] {event.event_type.upper()} on {event.camera_id}: {event.description}")

    # Find the global vehicle for this track
    all_vehicles = state.reid_engine.get_all_vehicles()
    gv = next(
        (v for v in all_vehicles
         if event.track_id in v.local_tracks.get(event.camera_id, [])),
        None
    )

    if gv is None:
        logger.warning(f"No GlobalVehicle found for track #{event.track_id} on {event.camera_id}")
        return

    # Predict escape routes
    routes = state.escape_router.predict_routes(event.camera_id, gv)
    state.orchestrator.apply_escape_routes(routes)

    # Generate report
    report = state.report_gen.generate(
        incident_type  = event.event_type,
        trigger_camera = event.camera_id,
        trigger_time   = event.timestamp,
        vehicle        = gv,
        escape_routes  = routes,
        graph          = state.graph,
        trigger_frame  = event.frame,
    )
    state.incidents.append(report)

    # Push alert to WS clients
    alert_payload = {
        "type":        "incident",
        "incident_id": report.incident_id,
        "event_type":  event.event_type,
        "camera_id":   event.camera_id,
        "location":    config.CAMERA_LOCATIONS.get(event.camera_id, event.camera_id),
        "timestamp":   event.timestamp.isoformat(),
        "description": event.description,
        "vehicle": {
            "global_id": gv.global_id,
            "plate":     gv.plate or "N/A",
            "is_motorcycle": gv.is_motorcycle,
        },
        "top_escape_route": routes[0].description if routes else "N/A",
    }
    await state.alert_queue.put(alert_payload)
    await _broadcast_alert(alert_payload)


# ══════════════════════════════════════════════════════════════════
# REST — CAMERAS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/cameras")
async def get_cameras():
    return state.stream_mgr.get_status()


@app.post("/api/cameras/start-all")
async def start_all_cameras():
    state.stream_mgr.start_all()
    return {"status": "ok", "message": "All cameras starting..."}


@app.post("/api/cameras/stop-all")
async def stop_all_cameras():
    state.stream_mgr.stop_all()
    return {"status": "ok", "message": "All cameras stopped."}


@app.post("/api/cameras/{camera_id}/start")
async def start_camera(camera_id: str):
    if camera_id not in config.RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found.")
    state.stream_mgr.start_camera(camera_id)
    return {"status": "ok", "camera_id": camera_id, "action": "started"}


@app.post("/api/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    if camera_id not in config.RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found.")
    state.stream_mgr.stop_camera(camera_id)
    return {"status": "ok", "camera_id": camera_id, "action": "stopped"}


@app.post("/api/cameras/{camera_id}/restart")
async def restart_camera(camera_id: str):
    if camera_id not in config.RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found.")
    state.stream_mgr.restart_camera(camera_id)
    return {"status": "ok", "camera_id": camera_id, "action": "restarted"}


# ══════════════════════════════════════════════════════════════════
# REST — INCIDENTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/incidents")
async def get_incidents():
    return [
        {
            "incident_id":   r.incident_id,
            "type":          r.incident_type,
            "camera_id":     r.trigger_camera,
            "location":      config.CAMERA_LOCATIONS.get(r.trigger_camera, r.trigger_camera),
            "time":          r.trigger_time.isoformat(),
            "vehicle_plate": r.vehicle.plate or "N/A",
            "generated_at":  r.generated_at.isoformat(),
        }
        for r in state.incidents
    ]


@app.get("/api/incidents/{incident_id}")
async def get_incident(incident_id: str):
    report = next((r for r in state.incidents if r.incident_id == incident_id), None)
    if report is None:
        raise HTTPException(404, f"Incident '{incident_id}' not found.")
    return report.to_dict()


@app.get("/api/incidents/{incident_id}/text")
async def get_incident_text(incident_id: str):
    report = next((r for r in state.incidents if r.incident_id == incident_id), None)
    if report is None:
        raise HTTPException(404, f"Incident '{incident_id}' not found.")
    return {"text": report.to_text()}


# ══════════════════════════════════════════════════════════════════
# REST — VEHICLES
# ══════════════════════════════════════════════════════════════════

@app.get("/api/vehicles")
async def get_vehicles():
    return state.reid_engine.summary()


@app.get("/api/vehicles/motorcycles")
async def get_motorcycles():
    motos = state.reid_engine.get_motorcycles()
    return [
        {
            "global_id": gv.global_id,
            "plate":     gv.plate or "N/A",
            "cameras":   gv.camera_count,
            "sightings": len(gv.sightings),
            "movement":  gv.movement_summary,
        }
        for gv in motos
    ]


@app.get("/api/vehicles/{global_id}/timeline")
async def get_vehicle_timeline(global_id: int):
    gv = state.reid_engine.get_vehicle(global_id)
    if gv is None:
        raise HTTPException(404, f"Vehicle #{global_id} not found.")
    return {
        "global_id":  gv.global_id,
        "plate":      gv.plate,
        "movement":   gv.movement_summary,
        "graph":      state.graph.get_timeline(gv.global_id),
    }


# ══════════════════════════════════════════════════════════════════
# REST — CAMERA PRIORITIES
# ══════════════════════════════════════════════════════════════════

@app.get("/api/priorities")
async def get_priorities():
    return state.orchestrator.get_all_priorities()


# ══════════════════════════════════════════════════════════════════
# WEBSOCKET — CAMERA STREAMS
# ══════════════════════════════════════════════════════════════════

@app.websocket("/ws/camera/{camera_id}")
async def camera_stream(websocket: WebSocket, camera_id: str):
    if camera_id not in config.RTSP_CAMERAS:
        await websocket.close(code=4004, reason="Camera not found")
        return

    await websocket.accept()
    state.camera_ws[camera_id].add(websocket)
    logger.info(f"WS client connected: {camera_id}  ({len(state.camera_ws[camera_id])} total)")

    try:
        while True:
            await websocket.receive_text()  # keep-alive ping
    except WebSocketDisconnect:
        state.camera_ws[camera_id].discard(websocket)
        logger.info(f"WS client disconnected: {camera_id}")


@app.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket):
    await websocket.accept()
    state.alert_ws.add(websocket)
    logger.info(f"Alert WS client connected  ({len(state.alert_ws)} total)")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        state.alert_ws.discard(websocket)


# ══════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════

@app.get("/")
async def serve_dashboard():
    p = Path("frontend/dashboard.html")
    if not p.exists():
        return JSONResponse({"status": "ok", "message": "Sentinel API running. Place dashboard.html in frontend/ folder."})
    return FileResponse(str(p))


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _encode_frame(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


async def _broadcast_frame(camera_id: str, jpg_b64: str):
    dead = set()
    for ws in state.camera_ws.get(camera_id, set()).copy():
        try:
            await ws.send_text(jpg_b64)
        except Exception:
            dead.add(ws)
    state.camera_ws[camera_id] -= dead


async def _broadcast_alert(payload: dict):
    import json
    dead = set()
    msg  = json.dumps(payload)
    for ws in state.alert_ws.copy():
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    state.alert_ws -= dead


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )