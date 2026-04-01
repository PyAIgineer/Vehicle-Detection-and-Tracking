"""
embedder.py — Vehicle Visual Embedding
=======================================
Extracts a 512-dim L2-normalised vector from a vehicle crop
using ResNet18 avgpool (torchvision).

Also extracts lightweight human-readable attributes
(dominant_color, helmet, clothing) via HSV heuristics —
these go into the Qdrant payload and graph nodes, not into the vector.

VehicleEmbedding
    .vector          np.ndarray  512-dim, L2-normalised  ← stored in Qdrant
    .dominant_color  str
    .helmet_present  bool
    .helmet_color    str
    .clothing_color  str
    .vehicle_type    str
    .plate_chars     str
"""

import cv2
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

import config

logger = logging.getLogger("Embedder")

# ── Color name lookup ──────────────────────────────────────────────
_COLOR_RANGES = [
    ((0,   15),  "red"),
    ((15,  35),  "orange"),
    ((35,  65),  "yellow"),
    ((65,  85),  "yellow-green"),
    ((85,  105), "green"),
    ((105, 130), "cyan"),
    ((130, 155), "blue"),
    ((155, 170), "purple"),
    ((170, 180), "red"),
]

def _hue_to_color(h: float, s: float, v: float) -> str:
    if s < 40:
        if v < 60:  return "black"
        if v < 140: return "grey"
        return "white"
    for (lo, hi), name in _COLOR_RANGES:
        if lo <= h < hi:
            return name
    return "red"


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════

@dataclass
class VehicleEmbedding:
    vector:         np.ndarray          # 512-dim, L2-normalised — stored in Qdrant
    dominant_color: str  = "unknown"    # payload / graph attribute
    helmet_present: bool = False
    helmet_color:   str  = "unknown"
    clothing_color: str  = "unknown"
    vehicle_type:   str  = "unknown"
    plate_chars:    str  = ""


# ══════════════════════════════════════════════════════════════════
# EMBEDDER
# ══════════════════════════════════════════════════════════════════

class VehicleEmbedder:
    """Load once, call extract() per detection crop."""

    def __init__(self):
        self._model     = None
        self._transform = None
        self._torch_ok  = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            import torchvision.models as tvm
            import torchvision.transforms as T

            backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
            self._model = torch.nn.Sequential(*list(backbone.children())[:-1])
            self._model.eval()
            self._model.to(config.EMBED_DEVICE)

            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((128, 64)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
            ])
            self._torch_ok = True
            logger.info("ResNet18 embedder ready.")
        except Exception as e:
            logger.warning(f"torch unavailable ({e}). Install torch for proper embeddings.")

    # ── Public ─────────────────────────────────────────────────────

    def extract(
        self,
        crop:         np.ndarray,
        vehicle_type: str = "unknown",
        plate_chars:  str = "",
    ) -> Optional[VehicleEmbedding]:
        if crop is None or crop.size == 0:
            return None
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            crop = cv2.resize(crop, (64, 128))

        hp = self._helmet_check(crop)
        return VehicleEmbedding(
            vector         = self._shape_vector(crop),
            dominant_color = self._dominant_color(crop),
            helmet_present = hp[0],
            helmet_color   = hp[1],
            clothing_color = self._clothing_color(crop),
            vehicle_type   = vehicle_type,
            plate_chars    = plate_chars,
        )

    # ── Shape vector ───────────────────────────────────────────────

    def _shape_vector(self, crop: np.ndarray) -> np.ndarray:
        if not self._torch_ok:
            rng = np.random.default_rng(abs(hash(crop.tobytes())) % (2**31))
            v   = rng.standard_normal(config.EMBED_DIM).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-8)
        try:
            import torch
            rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(config.EMBED_DEVICE)
            with torch.no_grad():
                feat = self._model(tensor).squeeze().cpu().numpy()
            return (feat / (np.linalg.norm(feat) + 1e-8)).astype(np.float32)
        except Exception as e:
            logger.debug(f"Shape vector failed: {e}")
            return np.zeros(config.EMBED_DIM, dtype=np.float32)

    # ── Attribute helpers ──────────────────────────────────────────

    def _dominant_color(self, crop: np.ndarray) -> str:
        try:
            hsv      = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            pixels   = hsv.reshape(-1, 3).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 3,
                                       cv2.KMEANS_RANDOM_CENTERS)
            h, s, v = centers[0]
            return _hue_to_color(float(h), float(s), float(v))
        except Exception:
            return "unknown"

    def _helmet_check(self, crop: np.ndarray) -> tuple[bool, str]:
        try:
            h  = crop.shape[0]
            y1 = int(h * config.HELMET_REGION_TOP)
            y2 = int(h * config.HELMET_REGION_BOT)
            region = crop[y1:y2]
            if region.size == 0:
                return False, "unknown"
            hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 30, 60), (180, 255, 255))
            if float(np.count_nonzero(mask)) / (mask.size + 1e-8) < config.HELMET_COVERAGE_THRESH:
                return False, "unknown"
            pixels = hsv[mask > 0].astype(np.float32)
            if len(pixels) == 0:
                return True, "unknown"
            mh, ms, mv = np.mean(pixels[:, 0]), np.mean(pixels[:, 1]), np.mean(pixels[:, 2])
            return True, _hue_to_color(float(mh), float(ms), float(mv))
        except Exception:
            return False, "unknown"

    def _clothing_color(self, crop: np.ndarray) -> str:
        try:
            h  = crop.shape[0]
            y1 = int(h * config.CLOTHING_REGION_TOP)
            y2 = int(h * config.CLOTHING_REGION_BOT)
            region = crop[y1:y2]
            return self._dominant_color(region) if region.size > 0 else "unknown"
        except Exception:
            return "unknown"