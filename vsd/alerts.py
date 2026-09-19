"""Alert layer: MQTT trigger (low latency) and optional Firebase mirror (durable, cloud)."""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class MqttPublisher:
    """Publishes `violations/{deviceId}` events. Never blocks or raises into the pipeline.

    `client` can be injected for tests; otherwise a paho-mqtt client connects in the
    background and keeps retrying, so a missing broker only means "no alerts sent yet".
    """

    def __init__(self, cfg: dict, device_id: str, client=None):
        self.topic = f"{cfg['topic_prefix']}/{device_id}"
        self.client = client
        if client is None:
            import paho.mqtt.client as mqtt

            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            if cfg.get("username"):
                self.client.username_pw_set(cfg["username"], cfg.get("password"))
            self.client.on_connect = lambda c, u, f, rc, p=None: log.info("MQTT connected (%s)", rc)
            self.client.on_disconnect = lambda c, u, f, rc, p=None: log.warning("MQTT disconnected (%s)", rc)
            self.client.reconnect_delay_set(1, 30)
            self.client.connect_async(cfg["host"], int(cfg["port"]))
            self.client.loop_start()

    def publish(self, vtype: str, confidence: float, ts: str, image_id: str) -> None:
        payload = json.dumps(
            {"type": vtype, "confidence": round(confidence, 2), "ts": ts, "imageId": image_id}
        )
        try:
            info = self.client.publish(self.topic, payload, qos=1)
            log.debug("MQTT -> %s %s (rc=%s)", self.topic, payload, getattr(info, "rc", "?"))
        except Exception:
            log.exception("MQTT publish failed")

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


class FirebaseSink:
    """Mirrors violations to Firestore + Storage from a background thread.

    Needs `pip install firebase-admin` and a service-account JSON; without them the
    pipeline simply runs on the local SQLite store.
    """

    def __init__(self, cfg: dict):
        import firebase_admin
        from firebase_admin import credentials, firestore, storage

        if not firebase_admin._apps:
            firebase_admin.initialize_app(
                credentials.Certificate(cfg["credentials"]),
                {"storageBucket": cfg["storage_bucket"]},
            )
        self._fs = firestore.client()
        self._bucket = storage.bucket()
        self._collection = cfg["collection"]
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="firebase").start()

    def push(self, record: dict, image_path: Path | None) -> None:
        self._q.put((record, image_path))

    def _run(self) -> None:
        while True:
            record, image_path = self._q.get()
            try:
                doc = {k: v for k, v in record.items() if k not in ("image", "full_image", "box")}
                if image_path and Path(image_path).exists():
                    blob = self._bucket.blob(f"violations/{record['id']}.jpg")
                    blob.upload_from_filename(str(image_path), content_type="image/jpeg")
                    blob.make_public()
                    doc["imageUrl"] = blob.public_url
                self._fs.collection(self._collection).document(record["id"]).set(doc)
            except Exception:
                log.exception("Firebase push failed for %s", record.get("id"))


def load_firebase(cfg: dict) -> FirebaseSink | None:
    if not cfg["enabled"]:
        return None
    try:
        return FirebaseSink(cfg)
    except ImportError:
        log.warning("firebase-admin not installed - Firebase mirroring disabled")
    except Exception:
        log.exception("Firebase init failed - Firebase mirroring disabled")
    return None
