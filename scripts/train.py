"""Train YOLOv8n, evaluate on the test split, check the mAP target, export to ONNX.

    python scripts/train.py --data data/helmet.yaml                 # phase 1: helmet classes
    python scripts/train.py --data data/full.yaml --name full       # phase 2: + seatbelt classes

The guide's command is `yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640`;
this wraps the same call, then reports per-class AP@0.5 against the >0.80 target for
helmet / no-helmet and copies the exported ONNX to models/helmet_seatbelt.onnx.
"""
import argparse
import shutil
from pathlib import Path

TARGET_MAP50 = 0.80
TARGET_CLASSES = ("helmet", "no-helmet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/helmet.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="lower it (8) if the GPU runs out of memory")
    ap.add_argument("--device", default="0", help="0 for the first GPU, cpu otherwise")
    ap.add_argument("--name", default="helmet")
    ap.add_argument("--out", default="models/helmet_seatbelt.onnx")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=a.device, project="runs", name=a.name, exist_ok=True)

    tuned = YOLO(str(model.trainer.best))  # Ultralytics nests output under runs/detect/
    m = tuned.val(data=a.data, split="test", imgsz=a.imgsz, device=a.device, plots=True)

    print("\nTest-split results")
    print(f"  mAP@0.5 (all classes): {m.box.map50:.3f}    mAP@0.5:0.95: {m.box.map:.3f}")
    print(f"  precision {m.box.mp:.3f}   recall {m.box.mr:.3f}")
    low = []
    for i, ap50 in zip(m.box.ap_class_index, m.box.ap50):
        name = tuned.names[int(i)]
        print(f"  {name:12s} AP@0.5 = {ap50:.3f}")
        if name in TARGET_CLASSES and ap50 < TARGET_MAP50:
            low.append(name)
    print("  confusion matrix saved to", Path(m.save_dir) / "confusion_matrix.png")
    if low:
        print(f"\n! {', '.join(low)} below the {TARGET_MAP50} target. Add more labelled local images "
              "(different lighting / angles) before tuning hyperparameters.")

    onnx = Path(tuned.export(format="onnx", imgsz=a.imgsz))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, a.out)
    print("exported ->", a.out)


if __name__ == "__main__":
    main()
