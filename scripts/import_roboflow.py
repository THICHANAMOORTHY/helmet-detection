"""Convert a Roboflow (or any YOLO-format) export into this project's classes.

    python scripts/import_roboflow.py --src downloads/bike-helmet-detection --dst datasets/raw
    python scripts/split_dataset.py  --src datasets/raw --dst datasets/helmet

Target classes: 0 rider, 1 helmet, 2 no-helmet (see data/helmet.yaml).

* Class names in the export's data.yaml are matched loosely ("With Helmet" -> helmet,
  "Without Helmet" -> no-helmet, "motorcyclist" -> rider). Anything unmatched (license plate,
  ...) is dropped. Override with --map "bike=rider".
* "bike" / "motorcycle" are NOT treated as rider by default: a box around only the bike
  won't contain the rider's head, which the rule engine needs. Use --bike-as-rider if the
  export's bike boxes do include the person, and check the containment report below.
* Roboflow augmentation writes several copies of one photo (name.rf.<hash>.jpg). Re-splitting
  those randomly would leak near-identical images between train and test, so only the first
  copy of each source photo is kept. Exporting with no augmentation avoids the issue entirely.
* Images whose boxes were all dropped are skipped (they'd become false "background").
"""
import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

TARGET = ["rider", "helmet", "no-helmet"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_MAP = {
    "helmet": "helmet", "withhelmet": "helmet", "wearinghelmet": "helmet", "helmeton": "helmet",
    "nohelmet": "no-helmet", "withouthelmet": "no-helmet", "nothelmet": "no-helmet",
    "helmetoff": "no-helmet", "nohelmets": "no-helmet",
    "rider": "rider", "motorcyclist": "rider", "bikerider": "rider", "motorcyclerider": "rider",
    "biker": "rider", "motorbikerider": "rider",
}
BIKE_NAMES = {"bike": "rider", "motorbike": "rider", "motorcycle": "rider", "motor": "rider"}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def build_map(names: list[str], extra: dict[str, str], bike_as_rider: bool) -> dict[int, int]:
    table = dict(DEFAULT_MAP)
    if bike_as_rider:
        table.update(BIKE_NAMES)
    table.update({norm(k): v for k, v in extra.items()})
    out = {}
    for i, n in enumerate(names):
        tgt = table.get(norm(n))
        if tgt:
            out[i] = TARGET.index(tgt)
    return out


def parse_line(line: str):
    """-> (class_id, cx, cy, w, h) from a detection line or a segmentation polygon line."""
    p = line.split()
    if len(p) < 5:
        return None
    c, v = int(float(p[0])), [float(x) for x in p[1:]]
    if len(v) == 4:
        return c, *v
    xs, ys = v[0::2], v[1::2]  # polygon -> bounding box
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def source_key(stem: str) -> str:
    return stem.split(".rf.")[0]


def inside(a, b) -> float:
    """Fraction of box a=(cx,cy,w,h) inside box b."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    return max(0, iw) * max(0, ih) / (a[2] * a[3]) if a[2] * a[3] > 0 else 0.0


def convert(src: Path, dst: Path, extra_map=None, bike_as_rider=False) -> dict:
    cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    cmap = build_map(names, extra_map or {}, bike_as_rider)
    if not cmap:
        raise SystemExit(f"No class in {names} matched rider/helmet/no-helmet. Use --map 'name=target'.")

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    seen, stats = set(), Counter()
    contained = total = 0

    for img in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT and "images" in p.parts):
        key = source_key(img.stem)
        if key in seen:
            stats["duplicate_copies_skipped"] += 1
            continue
        lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        rows = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                r = parse_line(line) if line.strip() else None
                if r and r[0] in cmap:
                    rows.append((cmap[r[0]], *r[1:]))
                elif r:
                    stats["boxes_dropped"] += 1
            if not rows and lbl.read_text().strip():
                stats["images_skipped_no_target_boxes"] += 1
                continue
        seen.add(key)
        shutil.copy2(img, dst / "images" / img.name)
        (dst / "labels" / f"{img.stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in rows))
        stats["images"] += 1
        for c, *_ in rows:
            stats[f"boxes_{TARGET[c]}"] += 1
        riders = [r[1:] for r in rows if r[0] == 0]
        for c, *box in rows:
            if c != 0:
                total += 1
                contained += any(inside(tuple(box), tuple(rd)) >= 0.5 for rd in riders)

    stats["heads_inside_a_rider_box"] = contained
    stats["heads_total"] = total
    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped Roboflow export (contains data.yaml)")
    ap.add_argument("--dst", default="datasets/raw")
    ap.add_argument("--map", action="append", default=[], metavar="NAME=TARGET",
                    help="extra class mapping, e.g. 'Person on bike=rider'")
    ap.add_argument("--bike-as-rider", action="store_true")
    a = ap.parse_args()
    extra = dict(m.split("=", 1) for m in a.map)
    s = convert(Path(a.src), Path(a.dst), extra, a.bike_as_rider)

    print(f"images kept: {s.get('images', 0)}")
    for k in TARGET:
        print(f"  {k:10s} boxes: {s.get(f'boxes_{k}', 0)}")
    for k in ("duplicate_copies_skipped", "images_skipped_no_target_boxes", "boxes_dropped"):
        if s.get(k):
            print(f"{k}: {s[k]}")
    if s["heads_total"]:
        frac = s["heads_inside_a_rider_box"] / s["heads_total"]
        print(f"helmet/no-helmet boxes inside a rider box: {frac:.0%}")
        if frac < 0.8:
            print("! Low. The rule engine only counts a no-helmet box that lies inside a rider box, so "
                  "this dataset's rider/bike boxes probably don't include the head. Pick another "
                  "dataset, or relabel rider boxes to cover the person.")


if __name__ == "__main__":
    main()
