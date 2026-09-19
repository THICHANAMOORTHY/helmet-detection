"""Dashboard web app: live feed, violation log, analytics, challans."""
from __future__ import annotations

import json
import queue
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from ..hub import EventBus, FrameHub
from ..storage import Store

STATIC = Path(__file__).parent / "static"


def create_app(store: Store, hub: FrameHub, bus: EventBus, snapshots: str | Path, info: dict) -> Flask:
    app = Flask(__name__, static_folder=None)
    snapshots = Path(snapshots).resolve()

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/info")
    def api_info():
        return jsonify(info)

    @app.get("/api/violations")
    def api_violations():
        return jsonify(store.list_violations(min(request.args.get("limit", 100, int), 500)))

    @app.get("/api/challans")
    def api_challans():
        return jsonify(store.list_challans(min(request.args.get("limit", 100, int), 500)))

    @app.get("/api/stats")
    def api_stats():
        return jsonify(store.stats(min(request.args.get("hours", 24, int), 24 * 30)))

    @app.get("/api/metrics")
    def api_metrics():
        return jsonify(hub.metrics)

    @app.get("/snapshots/<path:name>")
    def snapshot(name: str):
        if not (snapshots / name).resolve().is_relative_to(snapshots):
            abort(404)
        return send_from_directory(snapshots, name)

    @app.get("/video")
    def video():
        def gen():
            for jpeg in hub.frames():
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/events")
    def events():
        def gen():
            q = bus.subscribe()
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        ev = q.get(timeout=15)
                        yield f"data: {json.dumps(ev)}\n\n"
                    except queue.Empty:
                        yield ": keep-alive\n\n"
            finally:
                bus.unsubscribe(q)

        return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app
