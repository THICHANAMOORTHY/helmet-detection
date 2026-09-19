"""State shared between the pipeline threads and the dashboard."""
from __future__ import annotations

import queue
import threading


class FrameHub:
    """Holds the latest annotated JPEG for the live view, and rolling pipeline metrics."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self.viewers = 0
        self.metrics: dict = {
            "fps": 0.0, "stream_fps": 0.0, "infer_ms": 0.0, "latency_ms": 0.0,
            "frames": 0, "violations": 0, "simulated": False,
        }

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg, self._seq = jpeg, self._seq + 1
            self._cond.notify_all()

    def frames(self):
        """Generator of JPEG bytes for one MJPEG client."""
        self.viewers += 1
        seen = 0
        try:
            while True:
                with self._cond:
                    if not self._cond.wait_for(lambda: self._seq != seen, timeout=5):
                        continue
                    seen, jpeg = self._seq, self._jpeg
                yield jpeg
        finally:
            self.viewers -= 1


class EventBus:
    """Fan-out of small events (new violation) to Server-Sent-Events clients."""

    def __init__(self):
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=50)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass
