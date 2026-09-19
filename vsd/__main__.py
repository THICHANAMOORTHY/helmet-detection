"""Entry point:  python -m vsd [--simulate] [--source ...] [--config config.yaml]"""
from __future__ import annotations

import argparse
import logging
import threading

from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Helmet & seatbelt violation detection host")
    ap.add_argument("--config", help="YAML config overriding the defaults")
    ap.add_argument("--source", help="0 (webcam), http://<esp32>:81/stream, or a video file")
    ap.add_argument("--weights", help="model weights (.pt or .onnx)")
    ap.add_argument("--simulate", action="store_true", help="synthetic camera + detector, no weights needed")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--no-plate", action="store_true", help="skip number-plate OCR")
    ap.add_argument("--show", action="store_true", help="also open an OpenCV preview window (q quits)")
    ap.add_argument("--loop", action="store_true", help="loop a video file")
    ap.add_argument("--port", type=int)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    over: dict = {}
    if a.source: over["source"] = a.source
    if a.weights: over.setdefault("model", {})["weights"] = a.weights
    if a.no_mqtt: over.setdefault("mqtt", {})["enabled"] = False
    if a.no_plate: over.setdefault("plate", {})["enabled"] = False
    if a.no_dashboard: over.setdefault("dashboard", {})["enabled"] = False
    if a.port: over.setdefault("dashboard", {})["port"] = a.port
    if a.loop: over["loop"] = True
    cfg = load_config(a.config, over)

    # heavy imports after argument parsing so --help stays instant
    from .alerts import MqttPublisher, load_firebase
    from .hub import EventBus, FrameHub
    from .pipeline import Pipeline
    from .plate import load_plate_reader
    from .storage import Store

    store = Store(cfg["storage"]["db"])
    hub, bus = FrameHub(), EventBus()

    if a.simulate:
        from .sim import SimulatedDetector, SimulatedSource

        source = SimulatedSource()
        detector = SimulatedDetector(source)
    else:
        from .detector import YoloDetector
        from .sources import open_source

        detector = YoloDetector(cfg["model"])
        source = open_source(cfg["source"], loop=bool(cfg.get("loop", a.loop)))

    publisher = MqttPublisher(cfg["mqtt"], cfg["device_id"]) if cfg["mqtt"]["enabled"] else None
    pipe = Pipeline(cfg, source, detector, store, plate_reader=load_plate_reader(cfg["plate"]),
                    publisher=publisher, firebase=load_firebase(cfg["storage"]["firebase"]),
                    hub=hub, bus=bus, simulated=a.simulate, show=a.show)

    if cfg["dashboard"]["enabled"]:
        from .dashboard.app import create_app

        app = create_app(store, hub, bus, cfg["storage"]["snapshots"],
                         {"device_id": cfg["device_id"], "location": cfg["location"]})
        d = cfg["dashboard"]
        threading.Thread(target=lambda: app.run(d["host"], d["port"], threaded=True, use_reloader=False),
                         daemon=True, name="dashboard").start()
        logging.getLogger("vsd").info("Dashboard: http://localhost:%d", d["port"])

    try:
        pipe.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        if publisher:
            publisher.close()


if __name__ == "__main__":
    main()
