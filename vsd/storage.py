"""Local event store (SQLite). Firebase mirroring lives in alerts.py and is optional."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id TEXT PRIMARY KEY, type TEXT NOT NULL, confidence REAL, ts TEXT NOT NULL,
    device_id TEXT, location TEXT, image TEXT, full_image TEXT,
    plate TEXT, plate_conf REAL, simulated INTEGER DEFAULT 0, box TEXT
);
CREATE INDEX IF NOT EXISTS idx_violations_ts ON violations(ts);
CREATE TABLE IF NOT EXISTS challans (
    id TEXT PRIMARY KEY, violation_id TEXT, plate TEXT, type TEXT, ts TEXT,
    fine INTEGER, status TEXT
);
"""


class Store:
    def __init__(self, db_path: str | Path = ":memory:"):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)

    def add_violation(self, v: dict) -> None:
        row = {**v, "box": json.dumps(v.get("box")), "simulated": int(bool(v.get("simulated")))}
        cols = ["id", "type", "confidence", "ts", "device_id", "location", "image",
                "full_image", "plate", "plate_conf", "simulated", "box"]
        with self._lock, self._db:
            self._db.execute(
                f"INSERT OR REPLACE INTO violations ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [row.get(c) for c in cols],
            )

    def add_challan(self, c: dict) -> None:
        cols = ["id", "violation_id", "plate", "type", "ts", "fine", "status"]
        with self._lock, self._db:
            self._db.execute(
                f"INSERT OR REPLACE INTO challans ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [c.get(k) for k in cols],
            )

    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def list_violations(self, limit: int = 100) -> list[dict]:
        return self._rows("SELECT * FROM violations ORDER BY ts DESC, id DESC LIMIT ?", (limit,))

    def list_challans(self, limit: int = 100) -> list[dict]:
        return self._rows("SELECT * FROM challans ORDER BY ts DESC, id DESC LIMIT ?", (limit,))

    def stats(self, hours: int = 24) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        by_type = {
            r["type"]: r["n"]
            for r in self._rows(
                "SELECT type, COUNT(*) AS n FROM violations WHERE ts >= ? GROUP BY type", (since,)
            )
        }
        by_hour: dict[str, dict[str, int]] = {}
        for r in self._rows(
            "SELECT substr(ts, 1, 13) AS hour, type, COUNT(*) AS n FROM violations "
            "WHERE ts >= ? GROUP BY hour, type",
            (since,),
        ):
            by_hour.setdefault(r["hour"], {})[r["type"]] = r["n"]  # hour is "YYYY-MM-DDTHH" (UTC)
        return {
            "hours": hours,
            "total": sum(by_type.values()),
            "by_type": by_type,
            "by_hour": [{"hour": h, **counts} for h, counts in sorted(by_hour.items())],
            "challan_total": self._rows("SELECT COALESCE(SUM(fine), 0) AS s FROM challans")[0]["s"],
        }
