"""Violation rule engine.

The model only reports objects (rider / helmet / no-helmet / seatbelt / no-seatbelt);
this module decides what counts as a violation:

* helmet   - a no-helmet box lies inside a rider box and no more-confident helmet box
             sits on the same head.
* seatbelt - a no-seatbelt box lies inside the cabin region and no more-confident
             seatbelt box sits on the same person.
* a candidate must be seen in `min_consecutive` consecutive frames before it fires,
  which suppresses one-frame false positives (motion blur, occlusion).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .types import Box, Detection, Violation, center, iou, ioa


@dataclass
class Candidate:
    type: str
    box: Box
    conf: float


@dataclass
class _Track:
    id: int
    type: str
    box: Box
    hits: int
    conf_sum: float
    last_seen: float
    missed: int = 0
    fired: bool = False


class RuleEngine:
    def __init__(self, cfg: dict):
        self.enabled = set(cfg["enabled"])
        self.min_consecutive = int(cfg["min_consecutive"])
        self.max_gap = int(cfg["max_gap"])
        self.overlap = float(cfg["overlap"])
        self.head_iou = float(cfg["head_iou"])
        self.match_iou = float(cfg["match_iou"])
        self.cooldown_s = float(cfg["cooldown_s"])
        self.cabin_roi = cfg.get("cabin_roi")
        self._tracks: list[_Track] = []
        self._next_id = 1

    # -- per-frame candidate detection -------------------------------------------------
    def candidates(self, dets: list[Detection], shape: tuple[int, ...]) -> list[Candidate]:
        out: list[Candidate] = []
        if "no-helmet" in self.enabled:
            out += self._helmet_candidates(dets)
        if "no-seatbelt" in self.enabled:
            out += self._seatbelt_candidates(dets, shape)
        return out

    def _helmet_candidates(self, dets: list[Detection]) -> list[Candidate]:
        riders = [d for d in dets if d.cls == "rider"]
        helmets = [d for d in dets if d.cls == "helmet"]
        bare: dict[int, list[Detection]] = {}
        for nh in (d for d in dets if d.cls == "no-helmet"):
            if any(h.conf >= nh.conf and iou(h.box, nh.box) >= self.head_iou for h in helmets):
                continue  # the model also sees a helmet on this head, and is surer of it
            # a bare head belongs to the rider box that contains most of it
            best = max(range(len(riders)), key=lambda i: ioa(nh.box, riders[i].box), default=None)
            if best is not None and ioa(nh.box, riders[best].box) >= self.overlap:
                bare.setdefault(best, []).append(nh)
        return [
            Candidate("no-helmet", riders[i].box, max(h.conf for h in heads))
            for i, heads in bare.items()
        ]

    def _seatbelt_candidates(self, dets: list[Detection], shape: tuple[int, ...]) -> list[Candidate]:
        h, w = shape[:2]
        roi = None
        if self.cabin_roi:
            x1, y1, x2, y2 = self.cabin_roi
            roi = (x1 * w, y1 * h, x2 * w, y2 * h)
        belts = [d for d in dets if d.cls == "seatbelt"]
        out = []
        for nb in (d for d in dets if d.cls == "no-seatbelt"):
            cx, cy = center(nb.box)
            if roi and not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                continue
            if any(b.conf >= nb.conf and iou(b.box, nb.box) >= self.head_iou for b in belts):
                continue
            out.append(Candidate("no-seatbelt", nb.box, nb.conf))
        return out

    # -- temporal confirmation ---------------------------------------------------------
    def update(
        self, dets: list[Detection], shape: tuple[int, ...], now: float
    ) -> list[Violation]:
        """Feed one frame of detections; returns violations confirmed on this frame.

        `now` is a monotonic clock reading in seconds (used only for the cooldown).
        """
        matched: set[int] = set()
        confirmed: list[Violation] = []
        new_tracks: list[_Track] = []

        for c in sorted(self.candidates(dets, shape), key=lambda c: -c.conf):
            best, best_iou = None, self.match_iou
            for t in self._tracks:
                if t.type != c.type or t.id in matched:
                    continue
                v = iou(t.box, c.box)
                if v >= best_iou:
                    best, best_iou = t, v
            if best is None:
                best = _Track(self._next_id, c.type, c.box, 0, 0.0, now)
                self._next_id += 1
                new_tracks.append(best)
            matched.add(best.id)
            best.box, best.hits, best.missed, best.last_seen = c.box, best.hits + 1, 0, now
            best.conf_sum += c.conf
            if best.hits >= self.min_consecutive and not best.fired:
                best.fired = True
                confirmed.append(
                    Violation(
                        c.type,
                        best.conf_sum / best.hits,
                        c.box,
                        best.id,
                        datetime.now(timezone.utc),
                    )
                )

        for t in self._tracks:
            if t.id not in matched:
                t.missed += 1
        self._tracks = [
            t
            for t in self._tracks + new_tracks
            if (t.fired and now - t.last_seen <= self.cooldown_s)
            or (not t.fired and t.missed <= self.max_gap)
        ]
        return confirmed
