"""
config.py — Sentinel Incident Intelligence System
==================================================
Single source of truth for ALL parameters.
Edit only this file to tune any behaviour.
"""

from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════╗
# ║                   CAMERA CONFIGURATION                          ║
# ╚══════════════════════════════════════════════════════════════════╝

RTSP_CAMERAS: dict[str, str] = {
    "cctv_01": "rtsp://admin:Honey@123@192.168.1.10:554/1/2",   # Intersection North
    "cctv_02": "rtsp://admin:Honey@123@192.168.1.10:554/2/2",   # Highway Entry
    "cctv_03": "rtsp://admin:Honey@123@192.168.1.10:554/3/2",   # Market Road
    "cctv_04": "rtsp://admin:Honey@123@192.168.1.10:554/4/2",   # Ring Road East
    "cctv_05": "rtsp://admin:Honey@123@192.168.1.10:554/5/2",   # Ring Road West
    "cctv_06": "rtsp://admin:Honey@123@192.168.1.10:554/6/2",
   # Exit Point South
}

# Human-readable camera location labels (shown in UI + reports)
CAMERA_LOCATIONS: dict[str, str] = {
    "cctv_01": "Intersection North",
    "cctv_02": "Highway Entry",
    "cctv_03": "Market Road",
    "cctv_04": "Ring Road East",
    "cctv_05": "Ring Road West",
    "cctv_06": "Exit Point South",
}

# GPS coordinates of each camera (lat, lon) — used for escape-route mapping
CAMERA_COORDS: dict[str, tuple[float, float]] = {
    "cctv_01": (18.5204, 73.8567),
    "cctv_02": (18.5280, 73.8610),
    "cctv_03": (18.5150, 73.8520),
    "cctv_04": (18.5100, 73.8650),
    "cctv_05": (18.5100, 73.8490),
    "cctv_06": (18.5050, 73.8567),
}


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   RTSP STREAM SETTINGS                          ║
# ╚══════════════════════════════════════════════════════════════════╝

RTSP_TRANSPORT     = "tcp"          # "tcp" | "udp"  — TCP is stable for LAN
RTSP_RECONNECT_MS  = 5000           # ms between reconnect attempts
FRAME_QUEUE_SIZE   = 2              # max frames buffered per stream (latency vs safety)
STREAM_FPS_LIMIT   = 10            # cap read FPS to save CPU (0 = unlimited)


# ╔══════════════════════════════════════════════════════════════════╗
# ║              DETECTION (YOLO — PLATE MODEL ONLY)                ║
# ╚══════════════════════════════════════════════════════════════════╝

PLATE_WEIGHTS      = "weights/best.pt"           # your plate detector — only model needed
YOLO_CONF          = 0.45
YOLO_IMG_SIZE      = 640
SKIP_FRAMES        = 5             # run YOLO every Nth frame per camera
                                   # 5  = ~2fps detection at 10fps feed


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   BYTETRACK TRACKER SETTINGS                    ║
# ╚══════════════════════════════════════════════════════════════════╝

TRACKER_TYPE       = "bytetrack"   # ultralytics built-in tracker
TRACK_MAX_AGE      = 30            # frames to keep lost track alive
TRACK_MIN_HITS     = 2             # frames before track is confirmed
TRACK_IOU_THRESH   = 0.3


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   LPR PIPELINE                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

GROQ_MODEL         = "meta-llama/llama-4-scout-17b-16e-instruct"
OCR_BACKEND        = "easyocr"     # "easyocr" | "surya"
OCR_GPU            = True
OCR_CONF_THRESH    = 0.5
LPR_SKIP_FRAMES    = 10            # run LPR every Nth frame (heavier than detection)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   VISUAL RE-ID (Cross-Camera)                   ║
# ╚══════════════════════════════════════════════════════════════════╝

REID_MODEL         = "osnet_x0_25"       # lightweight ReID model
REID_THRESHOLD     = 0.65                # cosine similarity threshold (0–1)
REID_FEATURE_DIM   = 512
REID_QUEUE_SIZE    = 100                 # max embeddings kept in memory per camera
COLOR_HIST_BINS    = 32                  # bins for visual color histogram features


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   INCIDENT DETECTION                            ║
# ╚══════════════════════════════════════════════════════════════════╝

# Crash/anomaly detection thresholds
ANOMALY_SPEED_THRESH   = 80.0     # km/h proxy (pixels/frame × calibration)
ANOMALY_DIRECTION_FLIP = True     # detect sudden direction reversals
ANOMALY_OVERLAP_THRESH = 0.4      # IoU for collision detection
INCIDENT_COOLDOWN_SEC  = 10       # seconds before a new incident can be triggered

# Fleeing vehicle detection
FLEE_SPEED_MULTIPLIER  = 1.8      # vehicle moving Nx faster than area average
FLEE_LANE_VIOLATION    = True     # flag wrong-side-of-road movement
FLEE_CAMERA_PRIORITY_BOOST = 3    # priority score bump for cameras on escape route


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   ESCAPE ROUTE PREDICTION                       ║
# ╚══════════════════════════════════════════════════════════════════╝

CITY_GRAPH_PATH    = "data/city_road_graph.json"    # OSMnx exported graph (optional)
ESCAPE_MAX_ROUTES  = 3             # top-N routes to predict
ESCAPE_HORIZON_MIN = 10            # how far ahead (minutes) to project
CAMERA_SCAN_RADIUS = 0.8           # km radius to activate cameras on route


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   SPATIOTEMPORAL GRAPH                          ║
# ╚══════════════════════════════════════════════════════════════════╝

GAP_FILL_MAX_SEC   = 120           # max gap (seconds) to probabilistically fill
GAP_FILL_CONFIDENCE= 0.6           # minimum confidence for gap-filled nodes
TIMELINE_MAX_NODES = 500           # cap graph size per incident


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   REPORT GENERATION                             ║
# ╚══════════════════════════════════════════════════════════════════╝

REPORT_OUTPUT_DIR  = Path("reports")
CLIPS_OUTPUT_DIR   = Path("clips")
REPORT_GROQ_MODEL  = "meta-llama/llama-4-maverick-17b-128e-instruct"  # bigger model for reports
CLIP_PRE_SEC       = 5             # seconds of footage before event
CLIP_POST_SEC      = 10            # seconds of footage after event


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   API SERVER                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

API_HOST           = "0.0.0.0"
API_PORT           = 8000
WEBSOCKET_FPS      = 8             # frames/sec pushed to dashboard per camera
JPEG_QUALITY       = 75            # JPEG compression for WS frames (lower = faster)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   PATHS                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

OUTPUT_DIR         = Path("output")
LOG_LEVEL          = "INFO"
LOG_FORMAT         = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"