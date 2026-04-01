"""
lpr_video_test.py — Full LPR Pipeline on Video File
=====================================================
Pipeline:
    [1] YOLO          → detect plate bbox from frame
    [2] Crop + Prep   → crop plate region, preprocess
    [3] Groq Vision   → first-pass plate text read on crop
    [4] OCR           → EasyOCR / SuryaOCR refines text
    [5] Validate      → Indian plate regex check

Edit PARAMETERS block below, then:
    python lpr_video_test.py

Install:
    pip install ultralytics opencv-python groq easyocr numpy pillow python-dotenv
    pip install surya-ocr   # only if OCR_BACKEND = "surya"
"""

import cv2
import re
import os
import json
import base64
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from ultralytics import YOLO

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("lpr_test")


# ╔══════════════════════════════════════════════════════╗
# ║                   PARAMETERS                        ║
# ║         ← Edit everything in this block →           ║
# ╚══════════════════════════════════════════════════════╝

# ── Input ─────────────────────────────────────────────
VIDEO_PATH      = "input_video/video1.mp4"            # path to your test video file

# ── YOLO ──────────────────────────────────────────────
YOLO_WEIGHTS    = "weights/lpr_best.pt"     # your plate detector weights
YOLO_CONF       = 0.45                  # YOLO detection confidence threshold
YOLO_IMG_SIZE   = 640                   # inference image size

# ── Groq ──────────────────────────────────────────────
GROQ_MODEL      = "meta-llama/llama-4-scout-17b-16e-instruct"
SKIP_FRAMES     = 10                    # run YOLO+Groq every Nth frame
                                        # 5  → gate/parking (slow vehicles)
                                        # 10 → street cam
                                        # 20 → highway / parked

# ── OCR ───────────────────────────────────────────────
OCR_BACKEND     = "surya"             # "easyocr"  |  "surya"
OCR_GPU         = True                  # False if no GPU
OCR_CONF_THRESH = 0.5                   # min OCR confidence to prefer OCR over Groq text

# ── Output ────────────────────────────────────────────
SAVE_VIDEO      = False                  # save annotated output .mp4
SHOW_WINDOW     = True                  # False on headless / SSH server
OUTPUT_DIR = Path("output")

# ╚══════════════════════════════════════════════════════╝


# ── Internal ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PLATE_REGEX  = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')
BH_REGEX     = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$')


# ══════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════

def validate_indian_plate(text: str):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if PLATE_REGEX.match(cleaned): return cleaned
    if BH_REGEX.match(cleaned):    return cleaned
    fixed = cleaned.replace('0','O').replace('1','I').replace('8','B')
    if PLATE_REGEX.match(fixed):   return fixed
    return None


def preprocess(img: np.ndarray) -> np.ndarray:
    """Resize, denoise, sharpen plate crop for better OCR."""
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale  = 80 / max(gray.shape[0], 1)
    gray   = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray   = cv2.fastNlMeansDenoising(gray, h=10)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray   = cv2.filter2D(gray, -1, kernel)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def frame_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


# ══════════════════════════════════════════════════════
# STEP 1 — YOLO PLATE DETECTOR
# ══════════════════════════════════════════════════════

def load_yolo():
    if not Path(YOLO_WEIGHTS).exists():
        logger.error(f"YOLO weights not found: '{YOLO_WEIGHTS}' — check YOLO_WEIGHTS in PARAMETERS.")
        raise FileNotFoundError(YOLO_WEIGHTS)
    logger.info(f"Loading YOLO from {YOLO_WEIGHTS}")
    model = YOLO(YOLO_WEIGHTS)
    logger.info("YOLO ready.")
    return model


def yolo_detect(model, frame: np.ndarray) -> list[dict]:
    """
    Returns list of detected plates:
      [{"bbox": (x1,y1,x2,y2), "conf": float, "crop": np.ndarray}, ...]
    """
    results = model(frame, conf=YOLO_CONF, imgsz=YOLO_IMG_SIZE, verbose=False)[0]
    plates  = []
    h, w    = frame.shape[:2]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        plates.append({
            "bbox": (x1, y1, x2, y2),
            "conf": round(float(box.conf[0]), 3),
            "crop": crop,
        })
    return plates


# ══════════════════════════════════════════════════════
# STEP 3 — GROQ VISION (first-pass read on cropped plate)
# ══════════════════════════════════════════════════════

GROQ_PROMPT = """You are reading an Indian vehicle number plate image.

The image is already cropped to just the plate region.
Read the plate text carefully.

Respond ONLY in this exact JSON format (no markdown, no extra text):
{
  "plate_text": "MH12AB1234 or empty string if unreadable",
  "confidence": 0.0
}

Rules:
- plate_text uppercase, no spaces, no dashes
- confidence between 0.0 and 1.0
- If unreadable, return empty plate_text and confidence 0.0"""


def groq_read_plate(client: Groq, crop: np.ndarray) -> tuple[str, float]:
    """Send cropped plate to Groq, get text + confidence back."""
    try:
        b64 = frame_to_b64(crop)
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": GROQ_PROMPT}
                ]
            }],
            max_tokens=80,
            temperature=0.1,
        )
        raw  = res.choices[0].message.content.strip()
        raw  = re.sub(r'^```(?:json)?|```$', '', raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return data.get("plate_text", ""), float(data.get("confidence", 0.0))
    except Exception as e:
        logger.warning(f"Groq error: {e}")
        return "", 0.0


# ══════════════════════════════════════════════════════
# STEP 4 — OCR BACKENDS
# ══════════════════════════════════════════════════════

def load_ocr(backend: str):
    if backend == "surya":
        from surya.ocr import run_ocr
        from surya.model.detection.segformer    import load_model as load_det_model
        from surya.model.detection.segformer    import load_processor as load_det_processor
        from surya.model.recognition.model      import load_model as load_rec_model
        from surya.model.recognition.processor  import load_processor as load_rec_processor

        logger.info("Loading SuryaOCR...")
        det_model     = load_det_model()
        det_processor = load_det_processor()
        rec_model     = load_rec_model()
        rec_processor = load_rec_processor()
        logger.info("SuryaOCR ready.")

        def surya_read(img: np.ndarray) -> tuple[str, float]:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = run_ocr(
                [pil], [["en"]],
                det_model, det_processor,
                rec_model, rec_processor
            )
            if not res or not res[0].text_lines: return "", 0.0
            lines = res[0].text_lines
            text  = "".join([l.text for l in lines]).upper().replace(" ", "")
            conf  = float(np.mean([l.confidence for l in lines]))
            return text, round(conf, 3)

        return surya_read, "surya"

    else:
        import easyocr
        logger.info("Loading EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=OCR_GPU)

        def easy_read(img: np.ndarray) -> tuple[str, float]:
            results = reader.readtext(
                img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            if not results: return "", 0.0
            text = "".join([r[1] for r in results]).upper().replace(" ", "")
            conf = float(np.mean([r[2] for r in results]))
            return text, round(conf, 3)

        return easy_read, "easyocr"


# ══════════════════════════════════════════════════════
# DRAW OVERLAY
# ══════════════════════════════════════════════════════

def draw_overlay(frame, bbox, plate_text, conf, is_valid, groq_text, yolo_conf):
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0) if is_valid else (0, 165, 255)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Plate text label (filled background)
    label = f"{plate_text}  ({conf:.2f})"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Groq hint below box
    hint = f"Groq: {groq_text or 'N/A'}  |  YOLO: {yolo_conf:.2f}"
    cv2.putText(frame, hint, (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    return frame


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def run():
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found in .env")
        return
    if not Path(VIDEO_PATH).exists():
        logger.error(f"Video not found: '{VIDEO_PATH}' — update VIDEO_PATH in PARAMETERS.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"Cannot open: {VIDEO_PATH}")
        return

    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video  : {VIDEO_PATH}  |  {W}x{H} @ {fps}fps  |  {total} frames")
    logger.info(f"YOLO   : {YOLO_WEIGHTS}  conf={YOLO_CONF}")
    logger.info(f"OCR    : {OCR_BACKEND} (gpu={OCR_GPU})  conf_thresh={OCR_CONF_THRESH}")
    logger.info(f"Groq   : {GROQ_MODEL}  skip={SKIP_FRAMES}")

    # Load models
    yolo_model       = load_yolo()
    ocr_fn, ocr_name = load_ocr(OCR_BACKEND)
    groq_client      = Groq(api_key=GROQ_API_KEY)

    writer = None
    if SAVE_VIDEO:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / (Path(VIDEO_PATH).stem + "_lpr_output.mp4")
        writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        logger.info(f"Saving → {out_path}")

    frame_idx    = 0
    last_results = []       # list of last drawn detections (one per plate)
    detections   = []       # full log

    if SHOW_WINDOW:
        logger.info("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        display    = frame.copy()

        # ── Every Nth frame: run full pipeline ────
        if frame_idx % SKIP_FRAMES == 0:
            logger.info(f"Frame {frame_idx}/{total} → YOLO...")
            plates = yolo_detect(yolo_model, frame)

            if plates:
                last_results = []

                for p in plates:
                    bbox      = p["bbox"]
                    crop      = p["crop"]
                    yolo_conf = p["conf"]

                    # Step 2: preprocess crop
                    processed = preprocess(crop)

                    # Step 3: Groq reads the cropped plate
                    groq_text, groq_conf = groq_read_plate(groq_client, processed)

                    # Step 4: OCR refines
                    ocr_text, ocr_conf = ocr_fn(processed)

                    # Step 5: fuse — OCR wins if confident, else fall back to Groq
                    final     = ocr_text if ocr_conf >= OCR_CONF_THRESH else (groq_text or ocr_text)
                    validated = validate_indian_plate(final)
                    is_valid  = validated is not None
                    plate_out = validated or final
                    fused_conf = round((yolo_conf + ocr_conf) / 2, 3)

                    logger.info(
                        f"  YOLO={yolo_conf:.2f} | "
                        f"Groq='{groq_text}'({groq_conf:.2f}) | "
                        f"OCR({ocr_name})='{ocr_text}'({ocr_conf:.2f}) | "
                        f"→ '{plate_out}'  valid={is_valid}"
                    )

                    last_results.append({
                        "bbox":       bbox,
                        "plate":      plate_out,
                        "conf":       fused_conf,
                        "is_valid":   is_valid,
                        "groq_text":  groq_text,
                        "yolo_conf":  yolo_conf,
                    })

                    detections.append({
                        "frame":       frame_idx,
                        "plate":       plate_out,
                        "valid":       is_valid,
                        "confidence":  fused_conf,
                        "yolo_conf":   yolo_conf,
                        "groq_text":   groq_text,
                        "ocr_text":    ocr_text,
                        "ocr_conf":    ocr_conf,
                        "ocr_backend": ocr_name,
                        "timestamp":   datetime.now().isoformat(),
                    })
            else:
                logger.info(f"  No plate detected by YOLO.")
                last_results = []

        # ── Persist last detections on every frame ─
        for r in last_results:
            display = draw_overlay(
                display,
                r["bbox"], r["plate"], r["conf"],
                r["is_valid"], r["groq_text"], r["yolo_conf"]
            )

        # Status bar
        n_plates = len(last_results)
        cv2.putText(
            display,
            f"Frame {frame_idx}/{total}  |  YOLO: {YOLO_WEIGHTS}  |  OCR: {OCR_BACKEND}  |  skip: {SKIP_FRAMES}  |  plates: {n_plates}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

        if writer:
            writer.write(display)

        if SHOW_WINDOW:
            cv2.imshow("LPR Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit.")
                break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────
    valid  = [d for d in detections if d["valid"]]
    unique = set(d["plate"] for d in valid)

    print("\n" + "═" * 50)
    print(f"  Total detections  : {len(detections)}")
    print(f"  Valid plates      : {len(valid)}")
    print(f"  Unique plates     : {len(unique)}")
    if unique:
        print("  Plates found:")
        for p in sorted(unique):
            print(f"    → {p}")
    print("═" * 50)

    out_json = OUTPUT_DIR / (Path(VIDEO_PATH).stem + "_lpr_detections.json")
    with open(out_json, "w") as f:
        json.dump(detections, f, indent=2)
    logger.info(f"Detections saved → {out_json}")


if __name__ == "__main__":
    run()