"""Object detector wrapper around Ultralytics YOLO (.pt or exported .onnx)."""
from __future__ import annotations

import logging
from pathlib import Path

from .types import ALL_CLASSES, Detection

log = logging.getLogger(__name__)


class YoloDetector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # heavy import, keep it lazy

        weights = Path(cfg["weights"])
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights}. Train one with scripts/train.py, "
                "or try the pipeline without a model using --simulate."
            )
        self.model = YOLO(str(weights), task="detect")
        self.names: dict[int, str] = dict(self.model.names)
        self.imgsz = int(cfg["imgsz"])
        self.default_conf = float(cfg["conf"])
        self.class_conf = {k: float(v) for k, v in cfg.get("class_conf", {}).items()}
        self.device = self._pick_device(cfg["device"])

        known = set(self.names.values()) & set(ALL_CLASSES)
        if not known:
            log.warning(
                "Model classes %s contain none of %s - the rule engine will never fire.",
                sorted(self.names.values()), ALL_CLASSES,
            )
        log.info("Loaded %s on %s, classes: %s", weights.name, self.device, self.names)

    @staticmethod
    def _pick_device(want: str) -> str:
        if want != "auto":
            return want
        import torch

        return "0" if torch.cuda.is_available() else "cpu"

    def detect(self, frame) -> list[Detection]:
        floor = min([self.default_conf, *self.class_conf.values()])
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=floor, device=self.device, verbose=False
        )[0]
        out = []
        for b in result.boxes:
            name = self.names[int(b.cls)]
            conf = float(b.conf)
            if conf >= self.class_conf.get(name, self.default_conf):
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                out.append(Detection(name, conf, (x1, y1, x2, y2)))
        return out
