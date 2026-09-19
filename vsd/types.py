"""Shared data types and box geometry helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

HELMET_CLASSES = ("rider", "helmet", "no-helmet")
SEATBELT_CLASSES = ("seatbelt", "no-seatbelt")
ALL_CLASSES = HELMET_CLASSES + SEATBELT_CLASSES


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    box: Box


@dataclass
class Violation:
    type: str  # "no-helmet" | "no-seatbelt"
    confidence: float
    box: Box  # rider box (helmet) or occupant box (seatbelt)
    track_id: int
    ts: datetime  # UTC


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(a: Box, b: Box) -> float:
    """Fraction of box `a` that lies inside box `b`."""
    aa = area(a)
    return intersection(a, b) / aa if aa > 0 else 0.0


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def clip(b: Box, w: int, h: int) -> Box:
    return (max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3]))


def pad(b: Box, fx: float, fy: float, down: float = 0.0) -> Box:
    """Grow a box by fractions of its width/height; `down` extends only the bottom edge."""
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - w * fx, b[1] - h * fy, b[2] + w * fx, b[3] + h * (fy + down))
