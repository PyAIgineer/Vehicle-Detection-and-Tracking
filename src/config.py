"""
config.py — Sentinel Incident Intelligence System
==================================================
Single source of truth for ALL parameters.
Edit only this file to tune any behaviour.
"""

from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════╗
# ║                   DEV / PROD MODE                               ║
# ╚══════════════════════════════════════════════════════════════════╝




# ╔══════════════════════════════════════════════════════════════════╗
# ║                   CAMERA CONFIGURATION                          ║
# ╚══════════════════════════════════════════════════════════════════╝

RTSP_CAMERAS: dict[str, str] = {
    "cctv_01": "rtsp://admin:Honey@123@192.168.1.10:554/1/2",
    "cctv_02": "rtsp://admin:Honey@123@192.168.1.10:554/2/2",
    "cctv_03": "rtsp://admin:Honey@123@192.168.1.10:554/3/2",
    "cctv_04": "rtsp://admin:Honey@123@192.168.1.10:554/4/2",
    "cctv_05": "rtsp://admin:Honey@123@192.168.1.10:554/5/2",
    "cctv_06": "rtsp://admin:Honey@123@192.168.1.10:554/6/2",
}

CAMERA_LOCATIONS: dict[str, str] = {
    "cctv_01": "Parking Lot A - Entry",
    "cctv_02": "Parking Lot A - Exit",
    "cctv_03": "Parking Lot B - North",
    "cctv_04": "Parking Lot B - South",
    "cctv_05": "Corridor East",
    "cctv_06": "Corridor West",
}

# GPS coords — same premises so clustered (adjust to actual site)
CAMERA_COORDS: dict[str, tuple[float, float]] = {
    "cctv_01": (18.5204, 73.8567),
    "cctv_02": (18.5206, 73.8570),
    "cctv_03": (18.5208, 73.8565),
    "cctv_04": (18.5210, 73.8568),
    "cctv_05": (18.5205, 73.8572),
    "cctv_06": (18.5203, 73.8563),
}


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   RTSP STREAM SETTINGS                          ║
# ╚══════════════════════════════════════════════════════════════════╝

RTSP_TRANSPORT    = "tcp"
RTSP_RECONNECT_MS = 5000
FRAME_QUEUE_SIZE  = 2
STREAM_FPS_LIMIT  = 10


# ╔══════════════════════════════════════════════════════════════════╗
# ║              DETECTION (YOLO — PLATE MODEL ONLY)                ║
# ╚══════════════════════════════════════════════════════════════════╝

PLATE_WEIGHTS  = "weights/lpr_best.pt"
YOLO_CONF      = 0.45
YOLO_IMG_SIZE  = 640
SKIP_FRAMES    = 5
CROP_PAD_PX    = 8      # pixels to pad around plate bbox for vehicle crop


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   BYTETRACK TRACKER SETTINGS                    ║
# ╚══════════════════════════════════════════════════════════════════╝

TRACKER_TYPE     = "bytetrack"
TRACK_MAX_AGE    = 30
TRACK_MIN_HITS   = 2
TRACK_IOU_THRESH = 0.3


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   LPR PIPELINE                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

GROQ_MODEL      = "meta-llama/llama-4-scout-17b-16e-instruct"
OCR_BACKEND     = "easyocr"
OCR_GPU         = True
OCR_CONF_THRESH = 0.5
LPR_SKIP_FRAMES = 10


# ╔══════════════════════════════════════════════════════════════════╗
# ║                VEHICLE EMBEDDING (Visual ReID)                  ║
# ╚══════════════════════════════════════════════════════════════════╝

EMBED_DIM    = 512          # ResNet18 avgpool output dim
EMBED_DEVICE = "cpu"        # "cuda" if GPU available

# Body-region fractions of crop height for attribute extraction
HELMET_REGION_TOP      = 0.00
HELMET_REGION_BOT      = 0.45
CLOTHING_REGION_TOP    = 0.35
CLOTHING_REGION_BOT    = 0.75
HELMET_COVERAGE_THRESH = 0.20


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   QDRANT VECTOR DB (ReID)                       ║
# ╚══════════════════════════════════════════════════════════════════╝

QDRANT_HOST       = "localhost"
QDRANT_PORT       = 6333
QDRANT_COLLECTION = "vehicles"
QDRANT_IN_MEMORY  = True    # True = in-process (dev); False = persistent server

REID_THRESHOLD    = 0.80    # cosine score threshold for match (Qdrant returns 0–1)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   INCIDENT / ANOMALY DETECTION                  ║
# ╚══════════════════════════════════════════════════════════════════╝

ANOMALY_OVERLAP_THRESH = 0.4      # IoU threshold for crash detection
INCIDENT_COOLDOWN_SEC  = 10

# Dev mode flags — speed is still computed but no flee/wrong-way events emitted
FLEE_ENABLED           = False    # set True for production
ANOMALY_DIRECTION_FLIP = False    # no U-turn alerts in parking lot
FLEE_SPEED_MULTIPLIER  = 1.8
FLEE_CAMERA_PRIORITY_BOOST = 3


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   ESCAPE ROUTE PREDICTION                       ║
# ╚══════════════════════════════════════════════════════════════════╝

ESCAPE_MAX_ROUTES  = 3
ESCAPE_HORIZON_MIN = 10
CAMERA_SCAN_RADIUS = 0.8


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   SPATIOTEMPORAL GRAPH                          ║
# ╚══════════════════════════════════════════════════════════════════╝

GRAPH_GAP_FILL_ENABLED = False   # disabled in dev (no city blind spots)
GRAPH_MAX_EDGE_GAP_SEC = 300     # max gap to still draw an edge
GAP_FILL_CONFIDENCE    = 0.6
TIMELINE_MAX_NODES     = 500


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   REPORT GENERATION                             ║
# ╚══════════════════════════════════════════════════════════════════╝

REPORT_OUTPUT_DIR = Path("reports")
CLIPS_OUTPUT_DIR  = Path("clips")
REPORT_GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
CLIP_PRE_SEC      = 5
CLIP_POST_SEC     = 10


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   API SERVER                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

API_HOST      = "0.0.0.0"
API_PORT      = 8000
WEBSOCKET_FPS = 8
JPEG_QUALITY  = 75


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   PATHS                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

OUTPUT_DIR = Path("output")
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"