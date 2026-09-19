import copy
import json

import pytest

from vsd.alerts import MqttPublisher
from vsd.config import DEFAULTS, load_config
from vsd.pipeline import Pipeline
from vsd.plate import build_challan, normalize_plate, plate_region
from vsd.sim import SimulatedDetector, SimulatedSource
from vsd.storage import Store


@pytest.mark.parametrize("raw,want,valid", [
    ("MH12AB1234", "MH12AB1234", True),
    ("mh 12 ab 1234", "MH12AB1234", True),
    ("MH-12-AB-1234", "MH12AB1234", True),
    ("MHl2AB1234", "MH12AB1234", True),  # l/I mistaken for 1
    ("MH12A81234", "MH12AB1234", True),  # 8 mistaken for B
    ("KA05MN4821", "KA05MN4821", True),
    ("HELLO", "HELLO", False),
])
def test_normalize_plate(raw, want, valid):
    assert normalize_plate(raw) == (want, valid)


def test_plate_region_extends_below_a_seatbelt_box():
    box = (300, 200, 360, 280)
    _, _, _, y2 = plate_region(box, "no-seatbelt", (480, 640, 3))
    assert y2 > 280 + (280 - 200)
    assert plate_region(box, "no-helmet", (480, 640, 3))[3] < y2


def test_mqtt_payload_matches_the_guide():
    class FakeClient:
        sent = []

        def publish(self, topic, payload, qos=0):
            self.sent.append((topic, json.loads(payload)))

    c = FakeClient()
    MqttPublisher(DEFAULTS["mqtt"], "cam01", client=c).publish(
        "no-helmet", 0.8712, "2026-09-19T10:22:31Z", "v_00234")
    assert c.sent == [("violations/cam01", {
        "type": "no-helmet", "confidence": 0.87, "ts": "2026-09-19T10:22:31Z", "imageId": "v_00234"})]


def test_store_roundtrip_and_stats():
    s = Store()
    rec = {"id": "v_1", "type": "no-helmet", "confidence": 0.9, "ts": "2999-01-01T10:05:00Z",
           "device_id": "cam01", "location": "X", "image": "a.jpg", "full_image": "b.jpg",
           "plate": "MH12AB1234", "plate_conf": 0.9, "simulated": True, "box": [1, 2, 3, 4]}
    s.add_violation(rec)
    s.add_violation({**rec, "id": "v_2", "type": "no-seatbelt", "ts": "2999-01-01T10:40:00Z"})
    s.add_challan(build_challan(rec, DEFAULTS["fines"]))
    assert [v["id"] for v in s.list_violations()] == ["v_2", "v_1"]
    assert s.list_violations()[0]["simulated"] == 1
    ch = s.list_challans()[0]
    assert ch["fine"] == 1000 and ch["status"] == "MOCK-ISSUED" and ch["plate"] == "MH12AB1234"
    assert build_challan({**rec, "plate": None}, DEFAULTS["fines"])["status"].startswith("PLATE-UNREAD")


def test_load_config_merges_nested_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("rules:\n  min_consecutive: 5\nlocation: Gate 2\n")
    cfg = load_config(p)
    assert cfg["rules"]["min_consecutive"] == 5 and cfg["rules"]["overlap"] == 0.5
    assert cfg["location"] == "Gate 2"


def test_simulated_pipeline_end_to_end(tmp_path):
    """4 scenes (helmet ok, no-helmet, seatbelt ok, no-seatbelt) => exactly 2 violations,
    and the one-frame no-helmet flicker in scene 1 must not produce a third."""
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["snapshots"] = str(tmp_path / "snaps")

    class FakeMqtt:
        sent = []

        def publish(self, topic, payload, qos=0):
            self.sent.append((topic, json.loads(payload)))

    mq = FakeMqtt()
    src = SimulatedSource(realtime=False)
    store = Store()
    pipe = Pipeline(cfg, src, SimulatedDetector(src), store, plate_reader=None,
                    publisher=MqttPublisher(cfg["mqtt"], "cam01", client=mq), simulated=True)
    pipe.run(max_frames=4 * (12 + 48) + 5)
    pipe.worker.drain()

    got = sorted(v["type"] for v in store.list_violations())
    assert got == ["no-helmet", "no-seatbelt"]
    assert sorted(p["type"] for _, p in mq.sent) == got
    assert all(t == "violations/cam01" for t, _ in mq.sent)
    for v in store.list_violations():
        assert v["simulated"] == 1
        assert (tmp_path / "snaps" / v["image"]).stat().st_size > 0
        assert (tmp_path / "snaps" / v["full_image"]).stat().st_size > 0
    assert {c["plate"] for c in store.list_challans()} == {"UNREAD"}  # OCR disabled here
    assert pipe.hub.metrics["frames"] > 0
