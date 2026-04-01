import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
lpr_pipeline.py — LPR Pipeline (RTSP-adapted from lpr_video_detect.py)
=======================================================================
Pipeline:
  [Frame] -> [YOLO Plate Detect] -> [Crop + Preprocess]
          -> [Groq Vision]       -> [OCR refine]
          -> [Indian plate validate] -> [LPRResult]
"""

import os
import cv2
import re
import json
import base64
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

from groq import Groq
from ultralytics import YOLO

import config

logger = logging.getLogger("LPRPipeline")

PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')
BH_REGEX    = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$')

GROQ_PROMPT = """You are reading an Indian vehicle number plate image.
The image is already cropped to just the plate region.
Respond ONLY in this exact JSON format (no markdown):
{"plate_text": "MH12AB1234 or empty string", "confidence": 0.0}
Rules: uppercase, no spaces/dashes. confidence 0.0-1.0."""


def validate_indian_plate(text: str) -> Optional[str]:
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if PLATE_REGEX.match(cleaned): return cleaned
    if BH_REGEX.match(cleaned):    return cleaned
    fixed = cleaned.replace('0','O').replace('1','I').replace('8','B')
    if PLATE_REGEX.match(fixed):   return fixed
    return None


def preprocess_crop(img: np.ndarray) -> np.ndarray:
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


class LPRResult:
    def __init__(self, camera_id, track_id, plate, valid, conf, bbox,
                 groq_text, ocr_text, yolo_conf, timestamp):
        self.camera_id = camera_id
        self.track_id  = track_id
        self.plate     = plate
        self.valid     = valid
        self.conf      = conf
        self.bbox      = bbox
        self.groq_text = groq_text
        self.ocr_text  = ocr_text
        self.yolo_conf = yolo_conf
        self.timestamp = timestamp

    def to_dict(self):
        return {
            "camera_id": self.camera_id,
            "track_id":  self.track_id,
            "plate":     self.plate,
            "valid":     self.valid,
            "confidence":self.conf,
            "timestamp": self.timestamp.isoformat(),
        }


class LPRPipeline:
    """Per-camera LPR pipeline for live RTSP frames."""

    def __init__(self):
        self._plate_model = None
        self._groq_client = None
        self._ocr_fn      = None
        self._frame_counters: dict[str, int]  = defaultdict(int)
        self._last_results:   dict[str, list] = defaultdict(list)
        self._load_models()

    def _load_models(self):
        if Path(config.PLATE_WEIGHTS).exists():
            logger.info(f"Loading LPR YOLO: {config.PLATE_WEIGHTS}")
            self._plate_model = YOLO(config.PLATE_WEIGHTS)
        else:
            logger.warning(f"Plate weights not found: {config.PLATE_WEIGHTS}")

        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self._groq_client = Groq(api_key=api_key)

        self._ocr_fn = self._load_ocr()

    def _load_ocr(self):
        if config.OCR_BACKEND == "easyocr":
            try:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=config.OCR_GPU)
                def easy_read(img):
                    results = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    if not results: return "", 0.0
                    text = "".join([r[1] for r in results]).upper().replace(" ", "")
                    conf = float(np.mean([r[2] for r in results]))
                    return text, round(conf, 3)
                logger.info("EasyOCR loaded.")
                return easy_read
            except Exception as e:
                logger.warning(f"EasyOCR load failed: {e}")
        return None

    def process_frame(self, frame, camera_id, track_id=None, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()

        self._frame_counters[camera_id] += 1
        frame_idx = self._frame_counters[camera_id]

        if frame_idx % config.LPR_SKIP_FRAMES != 0 or self._plate_model is None:
            return self._last_results[camera_id]

        plates  = self._detect_plates(frame)
        results = []

        for p in plates:
            crop      = p["crop"]
            bbox      = p["bbox"]
            yolo_conf = p["conf"]
            processed = preprocess_crop(crop)

            groq_text, groq_conf = "", 0.0
            if self._groq_client:
                groq_text, groq_conf = self._groq_read(processed)

            ocr_text, ocr_conf = "", 0.0
            if self._ocr_fn:
                ocr_text, ocr_conf = self._ocr_fn(processed)

            final     = ocr_text if ocr_conf >= config.OCR_CONF_THRESH else (groq_text or ocr_text)
            validated = validate_indian_plate(final)
            is_valid  = validated is not None
            plate_out = validated or final
            fused_conf = round((yolo_conf + ocr_conf) / 2, 3)

            logger.info(
                f"[LPR][{camera_id}] YOLO={yolo_conf:.2f} | "
                f"Groq='{groq_text}' | OCR='{ocr_text}' | -> '{plate_out}' valid={is_valid}"
            )

            results.append(LPRResult(
                camera_id=camera_id, track_id=track_id,
                plate=plate_out, valid=is_valid, conf=fused_conf,
                bbox=bbox, groq_text=groq_text, ocr_text=ocr_text,
                yolo_conf=yolo_conf, timestamp=timestamp,
            ))

        self._last_results[camera_id] = results
        return results

    def _detect_plates(self, frame):
        results = self._plate_model(frame, conf=config.YOLO_CONF,
                                    imgsz=config.YOLO_IMG_SIZE, verbose=False)[0]
        plates = []
        h, w   = frame.shape[:2]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            plates.append({"bbox": (x1,y1,x2,y2), "conf": round(float(box.conf[0]),3), "crop": crop})
        return plates

    def _groq_read(self, crop):
        try:
            b64 = frame_to_b64(crop)
            res = self._groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role":"user","content":[
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
                    {"type":"text","text":GROQ_PROMPT}
                ]}],
                max_tokens=80, temperature=0.1,
            )
            raw  = res.choices[0].message.content.strip()
            raw  = re.sub(r'^```(?:json)?|```$', '', raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)
            return data.get("plate_text",""), float(data.get("confidence",0.0))
        except Exception as e:
            logger.warning(f"Groq LPR error: {e}")
            return "", 0.0

    def draw_overlay(self, frame, results):
        """Draw LPR results onto frame."""
        for r in results:
            x1, y1, x2, y2 = r.bbox
            color = (0,255,0) if r.valid else (0,165,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{r.plate} ({r.conf:.2f})"
            (lw,lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x1, y1-lh-10), (x1+lw+6, y1), color, -1)
            cv2.putText(frame, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
        return frame


# ══════════════════════════════════════════════════════════════════
# STANDALONE TEST RUNNER
# python lpr_pipeline.py                        # uses VIDEO_PATH from config
# python lpr_pipeline.py path/to/video.mp4
# python lpr_pipeline.py rtsp://admin:pass@...
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    load_dotenv()

    # ── Source: CLI arg > config VIDEO_PATH > webcam ──────────────
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        # fallback: try VIDEO_PATH from original lpr_video_detect style, else webcam
        source = getattr(config, "VIDEO_PATH", None) or 0

    print(f"\n{'═'*55}")
    print(f"  SENTINEL LPR — Standalone Test")
    print(f"  Source : {source}")
    print(f"  Models : plate={config.PLATE_WEIGHTS}  ocr={config.OCR_BACKEND}")
    print(f"{'═'*55}")
    print("  EasyOCR loading... (5-6 sec is normal first-time load)")

    pipeline = LPRPipeline()   # <-- EasyOCR loads here, pause is expected

    print("  Models ready. Opening source...\n")

    # ── Open capture ──────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)

    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_rtsp = isinstance(source, str) and source.startswith("rtsp")

    # ── Check if GUI display is available ────────────────────────
    USE_WINDOW = False
    try:
        test_frame = cv2.imread.__module__  # dummy — just test import
        cv2.namedWindow("__sentinel_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__sentinel_test__")
        USE_WINDOW = True
    except Exception:
        USE_WINDOW = False

    if USE_WINDOW:
        print(f"  Source open  |  fps={fps}  |  {'live stream' if is_rtsp else f'{total} frames'}")
        print("  Press Q in window to quit, or Ctrl+C in terminal.\n")
    else:
        print(f"  Source open  |  fps={fps}  |  {'live stream' if is_rtsp else f'{total} frames'}")
        print("  [No display] Running headless — detections logged to terminal + saved to output/")
        print("  Press Ctrl+C to stop.\n")

    # ── Output dir for saved annotated frames ─────────────────────
    save_dir = Path("output")
    save_dir.mkdir(exist_ok=True)
    writer = None
    if not USE_WINDOW:
        out_path = save_dir / "lpr_test_output.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = None  # init after first frame so we know W/H

    frame_idx     = 0
    all_detections = []
    t_start        = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            display    = frame.copy()

            # Run LPR (respects LPR_SKIP_FRAMES internally)
            results = pipeline.process_frame(frame, camera_id="test_cam", timestamp=datetime.now())

            # Draw plate overlays
            display = pipeline.draw_overlay(display, results)

            # Status bar
            elapsed    = time.time() - t_start
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            n_plates   = len(results)
            status = (
                f"Frame {frame_idx}"
                + (f"/{total}" if total > 0 else "")
                + f"  |  {fps_actual:.1f}fps  |  plates: {n_plates}"
                + (f"  |  skip: {config.LPR_SKIP_FRAMES}" if config.LPR_SKIP_FRAMES > 1 else "")
            )
            cv2.putText(display, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Log valid plates to terminal
            for r in results:
                if r.valid:
                    if r.plate not in all_detections:  # print only on first occurrence
                        print(f"  [Frame {frame_idx:04d}] ✔ {r.plate:<14} conf={r.conf:.2f}  "
                              f"groq='{r.groq_text}'  ocr='{r.ocr_text}'")
                    all_detections.append(r.plate)

            if USE_WINDOW:
                # Encode title safely for Windows (avoid unicode crash in Ultralytics patch)
                cv2.imshow("Sentinel LPR Test", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:   # Q or Escape
                    print("\n  [Quit]")
                    break
            else:
                # Headless: write annotated frames to MP4
                if writer is None:
                    h, w = display.shape[:2]
                    writer = cv2.VideoWriter(
                        str(save_dir / "lpr_test_output.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (w, h)
                    )
                writer.write(display)

                # Print progress every 50 frames
                if frame_idx % 50 == 0:
                    print(f"  [Frame {frame_idx:04d}] processed  fps={fps_actual:.1f}")

    except KeyboardInterrupt:
        print("\n  [Stopped by user]")

    cap.release()
    if writer:
        writer.release()
        print(f"  Annotated video saved → {save_dir}/lpr_test_output.mp4")
    if USE_WINDOW:
        cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────
    unique = sorted(set(all_detections))
    print(f"\n{'═'*55}")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Valid detections : {len(all_detections)}")
    print(f"  Unique plates    : {len(unique)}")
    if unique:
        for p in unique:
            print(f"    → {p}")
    print(f"{'═'*55}\n")