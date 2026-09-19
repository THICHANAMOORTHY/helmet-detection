"""Configuration: built-in defaults, overridden by an optional YAML file."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "device_id": "cam01",
    "location": "Junction A",
    # 0 = webcam, http://<esp32-ip>:81/stream = ESP32-CAM, or a video file path
    "source": "0",
    "model": {
        "weights": "models/helmet_seatbelt.onnx",
        "imgsz": 640,
        "conf": 0.5,  # default threshold
        "class_conf": {},  # per-class overrides, e.g. {"no-helmet": 0.55}
        "device": "auto",
    },
    "rules": {
        "enabled": ["no-helmet", "no-seatbelt"],
        "min_consecutive": 3,  # frames a violation must hold before firing
        "max_gap": 0,  # tolerated missed frames inside that run (0 = strictly consecutive)
        "overlap": 0.5,  # share of a no-helmet box that must lie inside a rider box
        "head_iou": 0.3,  # helmet/no-helmet boxes this similar are the same head
        "match_iou": 0.25,  # IoU used to follow one violation from frame to frame
        "cooldown_s": 10.0,  # a fired violation is not re-fired until it has been gone this long
        "cabin_roi": None,  # [x1, y1, x2, y2] normalised 0-1; None = whole frame
        "require_rider": True,  # if false, bare heads directly trigger violations without needing a rider box
    },
    "mqtt": {
        "enabled": True,
        "host": "localhost",
        "port": 1883,
        "username": None,
        "password": None,
        "topic_prefix": "violations",
    },
    "storage": {
        "db": "data/violations.db",
        "snapshots": "data/snapshots",
        "firebase": {
            "enabled": False,
            "credentials": "firebase-service-account.json",
            "storage_bucket": "",
            "collection": "violations",
        },
    },
    "plate": {
        "enabled": True,
        "weights": None,  # optional YOLO plate detector; else EasyOCR scans the region
        "languages": ["en"],
        "gpu": False,
    },
    "fines": {"no-helmet": 1000, "no-seatbelt": 1000},  # INR, mock values
    "dashboard": {"enabled": True, "host": "0.0.0.0", "port": 8000},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return deep_merge(cfg, overrides or {})
