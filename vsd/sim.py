"""Synthetic camera + detector for exercising the whole pipeline without model weights.

Draws simple bikes and cars crossing a road and reports the ground-truth boxes as if a
model had detected them. It exists to test the rule engine, alerts, storage and
dashboard end to end - it says nothing about real-world detection accuracy. Every
violation it produces is flagged `simulated` in the database and on the dashboard.
"""
from __future__ import annotations

import random
import time

import cv2
import numpy as np

from .types import Detection

W, H = 640, 480
IDLE = 12  # empty frames between scenes
SCENE = 48  # frames per scene

# (kind, wearing_safety_gear, plate text drawn on the vehicle)
SCENES = [
    ("bike", True, "MH12AB1234"),
    ("bike", False, "KA05MN4821"),
    ("car", True, "DL8CAF5030"),
    ("car", False, "TN09BX7712"),
]
FLICKER_FRAME = 20  # in scene 0, one frame wrongly reports no-helmet (must be debounced)


class SimulatedSource:
    ended = False

    def __init__(self, fps: float = 15.0, realtime: bool = True, seed: int = 7):
        self.fps = fps
        self.stream_fps = fps
        self.interval = 1.0 / fps if realtime else 0.0
        self.rng = random.Random(seed)
        self.frame_no = 0
        self.dets: list[Detection] = []
        self._next = time.perf_counter()

    def read(self, timeout: float = 1.0):
        wait = self._next - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next, time.perf_counter() - self.interval) + self.interval
        frame = self._render(self.frame_no)
        self.frame_no += 1
        return frame, time.perf_counter()

    def close(self) -> None:
        pass

    def _j(self, v: float, s: float = 3.0) -> float:
        return v + self.rng.uniform(-s, s)

    def _render(self, n: int) -> np.ndarray:
        img = np.full((H, W, 3), (70, 72, 74), np.uint8)
        cv2.rectangle(img, (0, 300), (W, 470), (52, 54, 56), -1)
        cv2.line(img, (0, 385), (W, 385), (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, "SIMULATED", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        self.dets = []
        cycle = IDLE + SCENE
        idx, k = divmod(n, cycle)
        if k < IDLE:
            return img
        k -= IDLE
        kind, safe, plate = SCENES[idx % len(SCENES)]
        x = int(-160 + (W + 320) * k / SCENE)  # left edge of the vehicle
        (self._bike if kind == "bike" else self._car)(img, x, safe, plate, k, idx % len(SCENES) == 0)
        return img

    def _plate(self, img, x, y, text):
        cv2.rectangle(img, (x, y), (x + 132, y + 32), (245, 245, 245), -1)
        cv2.putText(img, text, (x + 5, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 2, cv2.LINE_AA)

    def _bike(self, img, x, safe, plate, k, flicker_scene):
        y = 300
        cv2.rectangle(img, (x, y + 60), (x + 130, y + 95), (30, 30, 34), -1)  # bike body
        cv2.circle(img, (x + 25, y + 100), 22, (15, 15, 15), -1)
        cv2.circle(img, (x + 105, y + 100), 22, (15, 15, 15), -1)
        cv2.rectangle(img, (x + 40, y - 10), (x + 90, y + 62), (140, 70, 30), -1)  # rider torso
        head = (x + 65, y - 30)
        cv2.circle(img, head, 20, (40, 40, 200) if safe else (120, 160, 210), -1)  # helmet / bare head
        self._plate(img, x - 4, y + 118, plate)
        rider = (x - 6, y - 52, x + 138, y + 124)
        head_box = (head[0] - 20, head[1] - 20, head[0] + 20, head[1] + 20)
        self.dets.append(Detection("rider", self._j(0.9, 0.05), tuple(self._j(v) for v in rider)))
        wrong = flicker_scene and k == FLICKER_FRAME
        label = "helmet" if safe else "no-helmet"
        if wrong:
            label = "no-helmet"
        self.dets.append(Detection(label, self._j(0.85, 0.06), tuple(self._j(v, 2) for v in head_box)))

    def _car(self, img, x, safe, plate, k, _):
        y = 290
        cv2.rectangle(img, (x, y + 40), (x + 220, y + 120), (150, 40, 40), -1)  # body
        cv2.rectangle(img, (x + 40, y), (x + 180, y + 45), (150, 40, 40), -1)  # cabin
        cv2.rectangle(img, (x + 50, y + 8), (x + 170, y + 42), (190, 210, 220), -1)  # window
        cv2.circle(img, (x + 45, y + 122), 22, (15, 15, 15), -1)
        cv2.circle(img, (x + 175, y + 122), 22, (15, 15, 15), -1)
        cv2.circle(img, (x + 80, y + 22), 12, (120, 160, 210), -1)  # driver head
        cv2.rectangle(img, (x + 68, y + 32), (x + 96, y + 42), (60, 60, 60), -1)  # torso
        if safe:
            cv2.line(img, (x + 72, y + 34), (x + 94, y + 41), (0, 0, 0), 3)  # belt
        self._plate(img, x + 44, y + 92, plate)
        occupant = (x + 62, y + 6, x + 102, y + 44)
        self.dets.append(
            Detection("seatbelt" if safe else "no-seatbelt", self._j(0.83, 0.06),
                      tuple(self._j(v, 2) for v in occupant))
        )


class SimulatedDetector:
    """Returns the ground-truth detections of the frame most recently produced by `source`."""

    def __init__(self, source: SimulatedSource):
        self.source = source

    def detect(self, frame) -> list[Detection]:
        return list(self.source.dets)
