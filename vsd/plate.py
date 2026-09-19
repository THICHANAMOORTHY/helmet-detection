"""Number-plate reading and mock E-Challan records (the guide's E-Challan extension)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .types import Box, clip, pad

log = logging.getLogger(__name__)

# Indian format: SS DD [A-Z]{1,3} DDDD, e.g. MH12AB1234
_PLATE = re.compile(r"^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$")
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_ALPHA = {v: k for k, v in {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}.items()}


@dataclass
class PlateResult:
    text: str
    conf: float
    valid: bool  # matches the Indian plate format after OCR clean-up


def normalize_plate(raw: str) -> tuple[str, bool]:
    """Upper-case, strip separators, and repair look-alike characters by position."""
    s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    if _PLATE.match(s):
        return s, True
    for series in (1, 2, 3):
        if len(s) != 2 + 2 + series + 4:
            continue
        chars = list(s)
        for i in range(len(chars)):
            want_digit = 2 <= i < 4 or i >= 4 + series
            if want_digit:
                chars[i] = _TO_DIGIT.get(chars[i], chars[i])
            else:
                chars[i] = _TO_ALPHA.get(chars[i], chars[i])
        fixed = "".join(chars)
        if _PLATE.match(fixed):
            return fixed, True
    return s, False


def plate_region(box: Box, vtype: str, shape: tuple[int, ...]) -> Box:
    """Where to look for the plate, given the violation box.

    A rider box already spans the bike, so it is padded a little. A seatbelt box covers
    only the occupant, so the search extends well below it towards the bumper.
    """
    h, w = shape[:2]
    if vtype == "no-seatbelt":
        return clip(pad(box, 0.6, 0.2, down=1.8), w, h)
    return clip(pad(box, 0.15, 0.1), w, h)


class PlateReader:
    """EasyOCR-based reader. Optionally narrows the search with a YOLO plate detector."""

    def __init__(self, cfg: dict):
        import easyocr  # optional dependency

        self.ocr = easyocr.Reader(cfg["languages"], gpu=bool(cfg["gpu"]), verbose=False)
        self.detector = None
        if cfg.get("weights"):
            from ultralytics import YOLO

            self.detector = YOLO(cfg["weights"], task="detect")

    def read(self, frame, region: Box) -> PlateResult | None:
        x1, y1, x2, y2 = (int(v) for v in region)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crops = [crop]
        if self.detector is not None:
            r = self.detector.predict(crop, verbose=False)[0]
            if len(r.boxes):
                best = max(r.boxes, key=lambda b: float(b.conf))
                px1, py1, px2, py2 = (int(v) for v in best.xyxy[0])
                crops = [crop[max(0, py1 - 4):py2 + 4, max(0, px1 - 4):px2 + 4]]

        best: PlateResult | None = None
        for c in crops:
            for _, text, conf in self.ocr.readtext(c, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"):
                norm, valid = normalize_plate(text)
                if len(norm) < 6:
                    continue
                cand = PlateResult(norm, float(conf), valid)
                if best is None or (cand.valid, cand.conf) > (best.valid, best.conf):
                    best = cand
        return best


def load_plate_reader(cfg: dict) -> PlateReader | None:
    if not cfg["enabled"]:
        return None
    try:
        return PlateReader(cfg)
    except ImportError:
        log.warning("easyocr not installed - number-plate reading disabled (pip install easyocr)")
    except Exception:  # model download / init failure should not take the pipeline down
        log.exception("Plate reader failed to initialise - number-plate reading disabled")
    return None


def build_challan(violation: dict, fines: dict) -> dict:
    """A mock challan (proof of concept - no government API is contacted)."""
    plate = violation.get("plate")
    ts = datetime.now(timezone.utc)
    return {
        "id": f"CH-{ts:%Y%m%d}-{violation['id'][-6:].upper()}",
        "violation_id": violation["id"],
        "plate": plate or "UNREAD",
        "type": violation["type"],
        "ts": violation["ts"],
        "fine": int(fines.get(violation["type"], 0)),
        "status": "MOCK-ISSUED" if plate else "PLATE-UNREAD (manual review)",
    }
