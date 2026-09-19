"""Extract frames from recorded traffic footage for labelling (default 1 fps).

    python scripts/extract_frames.py footage/junction.mp4 --out datasets/raw/images --fps 1
"""
import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="datasets/raw/images")
    ap.add_argument("--fps", type=float, default=1.0, help="frames to keep per second of video")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in a.videos:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {v}")
        step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25.0) / a.fps))
        stem, i, kept = Path(v).stem, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                cv2.imwrite(str(out / f"{stem}_{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                kept += 1
            i += 1
        print(f"{v}: kept {kept} of {i} frames")


if __name__ == "__main__":
    main()
