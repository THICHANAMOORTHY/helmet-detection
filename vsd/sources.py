"""Frame sources. Each exposes read() -> (frame, arrival_time) | None, `ended`, close()."""
from __future__ import annotations

import logging
import threading
import time

import cv2

log = logging.getLogger(__name__)


class LiveSource:
    """Webcam or ESP32-CAM MJPEG stream.

    A reader thread always keeps only the newest frame, so a slow detector skips frames
    instead of falling behind (latency stays bounded). It reconnects if the stream drops.
    """

    ended = False

    def __init__(self, spec: str, reconnect_s: float = 2.0):
        self.spec = int(spec) if str(spec).isdigit() else spec
        self.reconnect_s = reconnect_s
        self.stream_fps = 0.0
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._seen = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="frame-reader").start()

    def _run(self) -> None:
        last = None
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.spec)
            if not cap.isOpened():
                log.warning("Cannot open %s, retrying in %.0fs", self.spec, self.reconnect_s)
                self._stop.wait(self.reconnect_s)
                continue
            log.info("Stream open: %s", self.spec)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.warning("Stream lost, reconnecting")
                    break
                now = time.perf_counter()
                if last is not None:
                    self.stream_fps = 0.9 * self.stream_fps + 0.1 / max(now - last, 1e-3)
                last = now
                with self._cond:
                    self._frame, self._t, self._seq = frame, now, self._seq + 1
                    self._cond.notify_all()
            cap.release()
            self._stop.wait(self.reconnect_s)

    def read(self, timeout: float = 1.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._seen, timeout):
                return None
            self._seen = self._seq
            return self._frame, self._t

    def close(self) -> None:
        self._stop.set()


class FileSource:
    """Recorded video. Paced to the file's own FPS so demos play at natural speed."""

    def __init__(self, path: str, realtime: bool = True, loop: bool = False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.interval = 1.0 / fps if realtime else 0.0
        self.loop = loop
        self.ended = False
        self.stream_fps = fps
        self._next = time.perf_counter()

    def read(self, timeout: float = 1.0):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            self.ended = True
            return None
        wait = self._next - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next, time.perf_counter() - self.interval) + self.interval
        return frame, time.perf_counter()

    def close(self) -> None:
        self.cap.release()


def open_source(spec: str, loop: bool = False):
    if str(spec).isdigit() or str(spec).startswith(("http://", "https://", "rtsp://")):
        return LiveSource(spec)
    return FileSource(spec, loop=loop)
