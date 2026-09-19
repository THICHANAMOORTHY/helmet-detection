import importlib.util
from pathlib import Path

import yaml

spec = importlib.util.spec_from_file_location(
    "import_roboflow", Path(__file__).resolve().parents[1] / "scripts" / "import_roboflow.py")
imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp)


def make_export(root: Path, names, files):
    (root).mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    for split, stem, label in files:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "images" / f"{stem}.jpg").write_bytes(b"x")
        (root / split / "labels" / f"{stem}.txt").write_text(label)


def test_remaps_drops_and_dedupes(tmp_path):
    names = ["motorcyclist", "With Helmet", "Without Helmet", "license_plate"]
    rider = "0 0.5 0.5 0.4 0.8"
    head_ok = "1 0.5 0.2 0.1 0.1"
    head_bad = "2 0.5 0.25 0.1 0.1"
    plate = "3 0.5 0.9 0.1 0.05"
    make_export(tmp_path / "exp", names, [
        ("train", "a_jpg.rf.111", f"{rider}\n{head_ok}\n{plate}"),
        ("train", "a_jpg.rf.222", f"{rider}\n{head_ok}"),  # augmented copy of a
        ("valid", "b_jpg.rf.333", f"{rider}\n{head_bad}"),
        ("valid", "c_jpg.rf.444", plate),  # only a dropped class -> skipped
    ])
    s = imp.convert(tmp_path / "exp", tmp_path / "out")
    assert s["images"] == 2
    assert s["duplicate_copies_skipped"] == 1
    assert s["images_skipped_no_target_boxes"] == 1
    assert s["boxes_dropped"] == 2  # the plate boxes in a and c
    assert (s["boxes_rider"], s["boxes_helmet"], s["boxes_no-helmet"]) == (2, 1, 1)
    assert s["heads_inside_a_rider_box"] == s["heads_total"] == 2
    lines = (tmp_path / "out" / "labels" / "b_jpg.rf.333.txt").read_text().split("\n")
    assert [ln.split()[0] for ln in lines] == ["0", "2"]  # remapped to rider, no-helmet


def test_bike_needs_opt_in_and_containment_is_reported(tmp_path):
    names = ["bike", "With Helmet", "Without Helmet"]
    bike_only = "0 0.5 0.8 0.5 0.2"  # box around the bike wheels; head is above it
    head = "2 0.5 0.2 0.1 0.1"
    make_export(tmp_path / "exp", names, [("train", "x", f"{bike_only}\n{head}")])
    s = imp.convert(tmp_path / "exp", tmp_path / "o1")
    assert s["boxes_no-helmet"] == 1 and s.get("boxes_rider", 0) == 0
    s = imp.convert(tmp_path / "exp", tmp_path / "o2", bike_as_rider=True)
    assert s["boxes_rider"] == 1
    assert s["heads_inside_a_rider_box"] == 0 and s["heads_total"] == 1  # would trigger the warning


def test_polygon_labels_become_boxes():
    c, x, y, w, h = imp.parse_line("1 0.2 0.2 0.4 0.2 0.4 0.6 0.2 0.6")
    assert (c, round(x, 2), round(y, 2), round(w, 2), round(h, 2)) == (1, 0.3, 0.4, 0.2, 0.4)
