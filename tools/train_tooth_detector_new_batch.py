#!/usr/bin/env python3
"""Train YOLO11n for a NEW detection batch.

Never writes to models/detection/tooth_detector_batch01/.
Never overwrites Batch 01 images or labels.

Example:
  python tools/train_tooth_detector_new_batch.py --batch 02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        required=True,
        help="Batch id without the word batch, e.g. 02 or 03",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional path to data.yaml. Default: fdi_detection_dataset/tooth_detector_batchNN/data.yaml "
        "or data/detection/batches/batchNN/data.yaml if that file exists.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    batch_id = args.batch.strip().lower().replace("batch", "")
    name = f"tooth_detector_batch{batch_id}"
    if name == "tooth_detector_batch01":
        raise SystemExit(
            "This script will not train Batch 01. "
            "Use tools/train_tooth_detector_batch01.py --force-retrain-batch01 if you really mean that."
        )

    out = ROOT / "models" / "detection" / name
    if out.exists() and (out / "weights" / "best.pt").exists():
        raise SystemExit(
            f"{out} already has weights/best.pt. Archive or rename it before training this batch again."
        )

    if args.data:
        data = Path(args.data)
    else:
        candidates = [
            ROOT / "data" / "detection" / "batches" / f"batch{batch_id}" / "data.yaml",
            ROOT / "fdi_detection_dataset" / f"tooth_detector_batch{batch_id}" / "data.yaml",
        ]
        data = next((p for p in candidates if p.exists()), None)
        if data is None:
            raise SystemExit(
                "No data.yaml found. Put a YOLO dataset at "
                f"data/detection/batches/batch{batch_id}/ (with data.yaml) "
                "or pass --data. See data/detection/README.md."
            )

    init = ROOT / "models" / "detection" / "pretrained" / "yolo11n.pt"
    model = YOLO(str(init) if init.exists() else "yolo11n.pt")
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=640,
        batch=8,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        cache=True,
        save=True,
        exist_ok=False,
        project=str(ROOT / "models" / "detection"),
        name=name,
        device="cpu",
        workers=0,
        plots=True,
        pretrained=True,
        val=True,
        seed=42,
        single_cls=True,
        verbose=True,
    )
    print("best", out / "weights" / "best.pt")


if __name__ == "__main__":
    main()
