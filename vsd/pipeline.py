"""Inference host: frames -> detector -> rule engine -> alert (MQTT) + record (DB/Firebase)."""
from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import timezone
from pathlib import Path

import cv2

from .alerts import MqttPublisher
from .hub import EventBus, FrameHub
from .plate import build_challan, plate_region
from .rules import RuleEngine
from .storage import Store
from .types import Detection, Violation, clip, pad

log = logging.getLogger(__name__)

_BAD = {"no-helmet", "no-seatbelt"}
_COLORS = {  # BGR
    "rider": (230, 160, 60), "helmet": (80, 200, 80), "seatbelt": (80, 200, 80),
    "no-helmet": (60, 60, 240), "no-seatbelt": (60, 60, 240),
}


def draw(frame, dets: list[Detection], hud: str, alerts: list[Violation]):
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        col = _COLORS.get(d.cls, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d.cls} {d.conf:.2f}", (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    for v in alerts:
        x1, y1, x2, y2 = (int(c) for c in v.box)
        cv2.rectangle(out, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 4)
        cv2.putText(out, f"VIOLATION: {v.type}", (x1, max(18, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, hud, (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class _EventWorker(threading.Thread):
    """Saves snapshots, reads the plate, writes records. Off the inference thread, so a slow
    OCR pass or network call never delays detection or the on-site alert."""

    def __init__(self, cfg, store: Store, plate_reader, firebase, bus: EventBus, hub: FrameHub):
        super().__init__(daemon=True, name="event-worker")
        self.cfg, self.store, self.reader, self.firebase = cfg, store, plate_reader, firebase
        self.bus, self.hub = bus, hub
        self.snap_dir = Path(cfg["storage"]["snapshots"])
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue = queue.Queue()

    def submit(self, job: dict) -> None:
        self.q.put(job)

    def drain(self, timeout: float = 30.0) -> None:
        """Block until queued jobs are done (used at shutdown and in tests)."""
        end = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < end:
            time.sleep(0.05)

    def run(self) -> None:
        while True:
            job = self.q.get()
            try:
                self._process(**job)
            except Exception:
                log.exception("Failed to record violation %s", job.get("vid"))
            finally:
                self.q.task_done()

    def _process(self, vid, v: Violation, frame, annotated, ts: str, simulated: bool) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(c) for c in clip(pad(v.box, 0.25, 0.25), w, h))
        crop_path = self.snap_dir / f"{vid}.jpg"
        cv2.imwrite(str(crop_path), annotated[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(self.snap_dir / f"{vid}_full.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

        plate = None
        if self.reader is not None:
            try:
                plate = self.reader.read(frame, plate_region(v.box, v.type, frame.shape))
            except Exception:
                log.exception("Plate read failed")

        rec = {
            "id": vid, "type": v.type, "confidence": round(v.confidence, 3), "ts": ts,
            "device_id": self.cfg["device_id"], "location": self.cfg["location"],
            "image": crop_path.name, "full_image": f"{vid}_full.jpg",
            "plate": plate.text if plate else None,
            "plate_conf": round(plate.conf, 3) if plate else None,
            "simulated": simulated, "box": [round(c, 1) for c in v.box],
        }
        self.store.add_violation(rec)
        self.store.add_challan(build_challan(rec, self.cfg["fines"]))
        if self.firebase:
            self.firebase.push(rec, crop_path)
        self.bus.publish({"event": "violation", "id": vid})
        log.info("Recorded %s %s plate=%s", vid, v.type, rec["plate"])


class Pipeline:
    def __init__(self, cfg, source, detector, store: Store, *, plate_reader=None,
                 publisher: MqttPublisher | None = None, firebase=None,
                 hub: FrameHub | None = None, bus: EventBus | None = None,
                 simulated: bool = False, show: bool = False):
        self.cfg, self.source, self.detector, self.store = cfg, source, detector, store
        self.publisher, self.simulated, self.show = publisher, simulated, show
        self.hub = hub or FrameHub()
        self.bus = bus or EventBus()
        self.rules = RuleEngine(cfg["rules"])
        self.worker = _EventWorker(cfg, store, plate_reader, firebase, self.bus, self.hub)
        self.worker.start()
        self.hub.metrics["simulated"] = simulated
        self._stop = threading.Event()
        self._n = 0
        self._recent: list[tuple[float, Violation]] = []  # (shown-until, violation) for overlay

    def stop(self) -> None:
        self._stop.set()

    def run(self, max_frames: int | None = None) -> None:
        m = self.hub.metrics
        t_last, frames_since = time.perf_counter(), 0
        try:
            while not self._stop.is_set() and (max_frames is None or m["frames"] < max_frames):
                item = self.source.read()
                if item is None:
                    if self.source.ended:
                        break
                    continue
                frame, t_in = item

                t0 = time.perf_counter()
                dets = self.detector.detect(frame)
                infer_ms = (time.perf_counter() - t0) * 1000
                new = self.rules.update(dets, frame.shape, time.monotonic())

                now = time.perf_counter()
                self._recent = [(t, v) for t, v in self._recent if t > now] + [(now + 1.0, v) for v in new]
                need_image = new or self.hub.viewers or self.show
                annotated = None
                if need_image:
                    hud = (f"{m['fps']:.1f} fps  infer {m['infer_ms']:.0f} ms  "
                           f"latency {m['latency_ms']:.0f} ms")
                    annotated = draw(frame, dets, hud, [v for _, v in self._recent])
                for v in new:
                    self._on_violation(v, frame, annotated)
                if annotated is not None:
                    if self.hub.viewers:
                        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            self.hub.publish(buf.tobytes())
                    if self.show:
                        cv2.imshow("Helmet & Seatbelt Violation Detection", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                m["frames"] += 1
                frames_since += 1
                m["infer_ms"] = 0.9 * m["infer_ms"] + 0.1 * infer_ms if m["infer_ms"] else infer_ms
                lat = (time.perf_counter() - t_in) * 1000
                m["latency_ms"] = 0.9 * m["latency_ms"] + 0.1 * lat if m["latency_ms"] else lat
                m["stream_fps"] = getattr(self.source, "stream_fps", 0.0)
                if now - t_last >= 2.0:
                    m["fps"] = frames_since / (now - t_last)
                    t_last, frames_since = now, 0
                    log.info("fps=%.1f stream_fps=%.1f infer=%.0fms e2e=%.0fms violations=%d",
                             m["fps"], m["stream_fps"], m["infer_ms"], m["latency_ms"], m["violations"])
        finally:
            self.worker.drain()
            self.source.close()
            if self.show:
                cv2.destroyAllWindows()

    def _on_violation(self, v: Violation, frame, annotated) -> None:
        self._n += 1
        ts = v.ts.astimezone(timezone.utc)
        vid = f"v_{ts:%Y%m%d%H%M%S}_{self._n:03d}"
        ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.hub.metrics["violations"] += 1
        log.warning("VIOLATION %s %s conf=%.2f", vid, v.type, v.confidence)
        if self.publisher:  # immediate: the on-site buzzer must not wait for OCR/DB
            self.publisher.publish(v.type, v.confidence, ts_iso, vid)
        self.worker.submit(dict(vid=vid, v=v, frame=frame.copy(), annotated=annotated.copy(),
                                ts=ts_iso, simulated=self.simulated))
