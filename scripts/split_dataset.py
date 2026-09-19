"""Split a YOLO-format dataset 70/20/10 (train/val/test), stratified by lighting.

Input layout (what Roboflow / LabelImg export):   <src>/images/*.jpg  <src>/labels/*.txt
Output layout (what data/*.yaml expects):         <dst>/images/{train,val,test}, <dst>/labels/{...}

Lighting bucket = mean brightness of the image (day / dusk / night), so every split gets
a similar mix of conditions. Images without a label file are treated as background images.

    python scripts/split_dataset.py --src datasets/raw --dst datasets/helmet
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

EXTS = {".jpg", ".jpeg", ".png"}


def lighting(path: Path) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "day"
    m = float(img.mean())
    return "night" if m < 60 else "dusk" if m < 110 else "day"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.2, 0.1))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    images = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"no images in {src / 'images'}")

    buckets = defaultdict(list)
    for p in images:
        buckets[lighting(p)].append(p)

    rng = random.Random(a.seed)
    splits = defaultdict(list)
    for name, items in sorted(buckets.items()):
        rng.shuffle(items)
        n_tr = round(len(items) * a.ratios[0])
        n_va = round(len(items) * a.ratios[1])
        splits["train"] += items[:n_tr]
        splits["val"] += items[n_tr:n_tr + n_va]
        splits["test"] += items[n_tr + n_va:]
        print(f"{name:6s}: {len(items)} images")

    for split, items in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p in items:
            shutil.copy2(p, dst / "images" / split / p.name)
            lbl = src / "labels" / f"{p.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst / "labels" / split / lbl.name)
        print(f"{split:5s}: {len(items)}")


if __name__ == "__main__":
    main()
