"""
main.py — Sentinel Incident Intelligence System — FastAPI Server (Dev Mode)
===========================================================================
Run:
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /api/cameras                     — list cameras + status
  POST /api/cameras/{id}/start          — start a camera stream
  POST /api/cameras/{id}/stop
  POST /api/cameras/start-all
  POST /api/cameras/stop-all
  GET  /api/incidents                   — list all incidents
  GET  /api/incidents/{id}              — full incident report
  GET  /api/vehicles                    — registry summary
  GET  /api/vehicles/{id}/timeline      — sightings + embedding + graph edges
  GET  /api/graph                       — full spatiotemporal graph
  GET  /api/graph/vehicles              — lightweight per-vehicle summary
  GET  /api/priorities                  — camera priority scores
  WS   /ws/camera/{id}                  — JPEG frame stream
  WS   /ws/alerts                       — real-time incident alerts
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import time
import asyncio
import base64
import logging
import numpy as np
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("SentinelAPI")

import config
from stream_manager  import StreamManager, StreamState
from detector        import VehicleDetector, IncidentEvent
from lpr_pipeline    import LPRPipeline
from reid            import ReIDEngine
from graph           import SpatiotemporalGraph
from escape_router   import EscapeRouter, CameraOrchestrator
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
        self.graph         = SpatiotemporalGraph()
        self.escape_router = EscapeRouter()
        self.orchestrator  = CameraOrchestrator(list(config.RTSP_CAMERAS.keys()))
        self.report_gen    = ReportGenerator()

        self.incidents:   list[IncidentReport] = []
        self.alert_queue: asyncio.Queue        = asyncio.Queue()

        self.bw_bytes:    dict[str, int]   = {c: 0 for c in config.RTSP_CAMERAS}
        self.bw_mbps:     dict[str, float] = {c: 0.0 for c in config.RTSP_CAMERAS}
        self.bw_reset_ts: float            = time.time()

        self.camera_ws: dict[str, set[WebSocket]] = {c: set() for c in config.RTSP_CAMERAS}
        self.alert_ws:  set[WebSocket]             = set()

        import secrets
        self.ws_token: str = secrets.token_urlsafe(32)
        logger.info("App state initialised.")


state: Optional[AppState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    state = AppState()
    logger.info("Sentinel started.")
    asyncio.create_task(pipeline_loop())
    yield
    state.stream_mgr.stop_all()
    logger.info("Sentinel shutdown.")


app = FastAPI(
    title   = "Sentinel Incident Intelligence System",
    version = "2.0.0-dev",
    lifespan= lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent   # D:\Vehicle_dt\
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ══════════════════════════════════════════════════════════════════
# PIPELINE LOOP
# ══════════════════════════════════════════════════════════════════

async def pipeline_loop():
    ws_delay = 1.0 / config.WEBSOCKET_FPS

    while True:
        frame_start = time.monotonic()

        for cam_id in state.stream_mgr.get_active_camera_ids():
            frame, ts = state.stream_mgr.get_frame(cam_id)
            if frame is None:
                continue

            # ── Detection + ByteTrack ───────────────────────────
            try:
                detections, annotated = state.detector.process_frame(frame, cam_id, ts)

                for det in detections:
                    # det.crop is already extracted (padded) in detector.py
                    # Run LPR on it for plate text
                    lpr_results = state.lpr.process_frame(
                        det.crop, cam_id, det.track_id, ts
                    )
                    plate_str   = None
                    plate_chars = ""
                    if lpr_results:
                        best = lpr_results[0]
                        if best.valid:
                            plate_str = best.plate
                        plate_chars = best.plate or ""
                        annotated   = state.lpr.draw_overlay(annotated, lpr_results)

                    # ── ReID ─────────────────────────────────────────
                    gv = state.reid_engine.match_or_create(
                        camera_id     = cam_id,
                        track_id      = det.track_id,
                        crop          = det.crop,
                        plate         = plate_str,
                        is_motorcycle = det.is_motorcycle,
                        timestamp     = ts,
                        bbox          = det.bbox,
                        plate_chars   = plate_chars,
                    )

                    # ── Spatiotemporal graph ──────────────────────────
                    state.graph.add_sighting(
                        global_vehicle_id = gv.global_id,
                        camera_id         = cam_id,
                        timestamp         = ts or datetime.now(),
                        confidence        = gv.sightings[-1].confidence,
                        plate             = gv.plate,
                        embedding         = gv.embedding,
                    )
            except Exception as exc:
                logger.error(f"[pipeline] {cam_id} frame error: {exc}", exc_info=True)
                annotated = frame   # push raw frame so stream stays alive

            # ── Push frame to WS clients ─────────────────────────
            if state.camera_ws.get(cam_id):
                jpg = _encode_frame(annotated)
                await _broadcast_frame(cam_id, jpg)

        # ── Process incidents ───────────────────────────────────
        for event in state.detector.pop_events():
            await _handle_incident(event)

        elapsed = time.monotonic() - frame_start
        await asyncio.sleep(max(0, ws_delay - elapsed))


async def _handle_incident(event: IncidentEvent):
    logger.warning(f"[INCIDENT] {event.event_type.upper()} @ {event.camera_id}: {event.description}")

    gv = next(
        (v for v in state.reid_engine.get_all_vehicles()
         if event.track_id in v.local_tracks.get(event.camera_id, [])),
        None,
    )
    if gv is None:
        logger.warning(f"No GlobalVehicle for track #{event.track_id} on {event.camera_id}")
        return

    routes = state.escape_router.predict_routes(event.camera_id, gv)
    state.orchestrator.apply_escape_routes(routes)

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

    emb = gv.embedding
    alert = {
        "type":        "incident",
        "incident_id": report.incident_id,
        "event_type":  event.event_type,
        "camera_id":   event.camera_id,
        "location":    config.CAMERA_LOCATIONS.get(event.camera_id, event.camera_id),
        "timestamp":   event.timestamp.isoformat(),
        "description": event.description,
        "vehicle": {
            "global_id":      gv.global_id,
            "plate":          gv.plate or "N/A",
            "is_motorcycle":  gv.is_motorcycle,
            "vehicle_color":  emb.dominant_color if emb else "unknown",
            "helmet_present": emb.helmet_present if emb else False,
            "clothing_color": emb.clothing_color if emb else "unknown",
        },
        "top_route": routes[0].description if routes else "N/A",
    }
    await _broadcast_alert(alert)


# ══════════════════════════════════════════════════════════════════
# REST — CAMERAS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/cameras")
async def get_cameras():
    return state.stream_mgr.get_status()


@app.post("/api/cameras/start-all")
async def start_all():
    state.stream_mgr.start_all()
    return {"status": "ok"}


@app.post("/api/cameras/stop-all")
async def stop_all():
    state.stream_mgr.stop_all()
    return {"status": "ok"}


@app.post("/api/cameras/{camera_id}/start")
async def start_camera(camera_id: str):
    if camera_id not in config.RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found.")
    state.stream_mgr.start_camera(camera_id)
    return {"status": "ok", "camera_id": camera_id}


@app.post("/api/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    if camera_id not in config.RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found.")
    state.stream_mgr.stop_camera(camera_id)
    return {"status": "ok", "camera_id": camera_id}


@app.post("/api/cameras/{camera_id}/restart")
async def restart_camera(camera_id: str):
    if camera_id not in config.RTSP_CAMERAS:
        raise HTTPException(404, f"Camera '{camera_id}' not found.")
    state.stream_mgr.restart_camera(camera_id)
    return {"status": "ok", "camera_id": camera_id}


# ══════════════════════════════════════════════════════════════════
# REST — INCIDENTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/incidents")
async def get_incidents():
    return [
        {
            "incident_id":    r.incident_id,
            "type":           r.incident_type,
            "camera_id":      r.trigger_camera,
            "location":       config.CAMERA_LOCATIONS.get(r.trigger_camera, r.trigger_camera),
            "time":           r.trigger_time.isoformat(),
            "vehicle_plate":  r.vehicle.plate or "N/A",
            "vehicle_color":  r.vehicle.embedding.dominant_color if r.vehicle.embedding else "unknown",
            "generated_at":   r.generated_at.isoformat(),
        }
        for r in state.incidents
    ]


@app.get("/api/incidents/{incident_id}")
async def get_incident(incident_id: str):
    report = next((r for r in state.incidents if r.incident_id == incident_id), None)
    if report is None:
        raise HTTPException(404, f"Incident '{incident_id}' not found.")
    return report.to_dict()


# ══════════════════════════════════════════════════════════════════
# REST — VEHICLES
# ══════════════════════════════════════════════════════════════════

@app.get("/api/vehicles")
async def get_vehicles():
    return state.reid_engine.summary()


@app.get("/api/vehicles/motorcycles")
async def get_motorcycles():
    return [
        {
            "global_id": gv.global_id,
            "plate":     gv.plate or "N/A",
            "cameras":   gv.camera_count,
            "sightings": len(gv.sightings),
            "movement":  gv.movement_summary,
        }
        for gv in state.reid_engine.get_motorcycles()
    ]


@app.get("/api/vehicles/{global_id}/timeline")
async def get_vehicle_timeline(global_id: int):
    gv = state.reid_engine.get_vehicle(global_id)
    if gv is None:
        raise HTTPException(404, f"Vehicle #{global_id} not found.")
    emb = gv.embedding
    return {
        "global_id":      gv.global_id,
        "plate":          gv.plate,
        "is_motorcycle":  gv.is_motorcycle,
        "embedding": {
            "dominant_color": emb.dominant_color if emb else None,
            "helmet_present": emb.helmet_present if emb else None,
            "helmet_color":   emb.helmet_color   if emb else None,
            "clothing_color": emb.clothing_color if emb else None,
            "vehicle_type":   emb.vehicle_type   if emb else None,
            "plate_chars":    emb.plate_chars     if emb else None,
        },
        "movement":   gv.movement_summary,
        "trajectory": state.graph.get_trajectory(gv.global_id),
        "edges":      state.graph.get_edges_for_vehicle(gv.global_id),
    }


# ══════════════════════════════════════════════════════════════════
# REST — SPATIOTEMPORAL GRAPH
# ══════════════════════════════════════════════════════════════════

@app.get("/api/graph")
async def get_graph():
    """Full graph — all nodes + edges."""
    return state.graph.to_dict()


@app.get("/api/graph/vehicles")
async def get_graph_vehicles():
    """Per-vehicle summary from graph (color, helmet, camera count…)."""
    return state.graph.get_all_vehicles_summary()


# ══════════════════════════════════════════════════════════════════
# REST — MISC
# ══════════════════════════════════════════════════════════════════

@app.get("/api/priorities")
async def get_priorities():
    return state.orchestrator.get_all_priorities()


@app.get("/api/token")
async def get_ws_token():
    return {"token": state.ws_token}


@app.get("/api/bandwidth")
async def get_bandwidth():
    result = {}
    for cam_id in config.RTSP_CAMERAS:
        mbps = state.bw_mbps.get(cam_id, 0.0)
        result[cam_id] = {
            "mbps":          mbps,
            "kb_per_frame":  round((mbps * 1_000_000 / 8) / max(config.WEBSOCKET_FPS, 1) / 1024, 1),
        }
    return {"cameras": result, "total_mbps": round(sum(v["mbps"] for v in result.values()), 2)}


# ══════════════════════════════════════════════════════════════════
# WEBSOCKET — CAMERA STREAMS
# ══════════════════════════════════════════════════════════════════

def _verify_token(token: Optional[str]) -> bool:
    import hmac
    return bool(token) and hmac.compare_digest(token, state.ws_token)


@app.websocket("/ws/camera/{camera_id}")
async def camera_stream(websocket: WebSocket, camera_id: str, token: Optional[str] = None):
    if camera_id not in config.RTSP_CAMERAS:
        await websocket.close(code=4004, reason="Camera not found")
        return
    if not _verify_token(token):
        await websocket.close(code=4003, reason="Unauthorized")
        return
    await websocket.accept()
    state.camera_ws[camera_id].add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        state.camera_ws[camera_id].discard(websocket)


@app.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket, token: Optional[str] = None):
    if not _verify_token(token):
        await websocket.close(code=4003, reason="Unauthorized")
        return
    await websocket.accept()
    state.alert_ws.add(websocket)
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
    p = FRONTEND_DIR / "dashboard.html"
    if not p.exists():
        return JSONResponse({"status": "ok", "message": "Sentinel API running. Add dashboard.html to frontend/."})
    return FileResponse(str(p))


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


async def _broadcast_frame(camera_id: str, jpg_b64: str):
    dead     = set()
    byte_len = len(jpg_b64)
    for ws in state.camera_ws.get(camera_id, set()).copy():
        try:
            await ws.send_text(jpg_b64)
            state.bw_bytes[camera_id] = state.bw_bytes.get(camera_id, 0) + byte_len
        except Exception:
            dead.add(ws)
    state.camera_ws[camera_id] -= dead

    now     = time.time()
    elapsed = now - state.bw_reset_ts
    if elapsed >= 2.0:
        for cid in config.RTSP_CAMERAS:
            bits = state.bw_bytes.get(cid, 0) * 8
            state.bw_mbps[cid]  = round(bits / elapsed / 1_000_000, 2)
            state.bw_bytes[cid] = 0
        state.bw_reset_ts = now


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
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT,
                reload=False, log_level=config.LOG_LEVEL.lower())