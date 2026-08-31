#!/usr/bin/env python3
"""Train YOLO11n on verified Batch_01 only. Never uses predictions/ or Batch_02 labels."""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fdi_detection_dataset" / "tooth_detector_batch01" / "data.yaml"
OUT = ROOT / "models" / "detection" / "tooth_detector_batch01"
ARCHIVES = ROOT / "archive" / "experiments" / "yolo_run_archives"
PRIOR_BEST = OUT / "weights" / "best.pt"


def verify() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "verify_tooth_detector_batch01.py")],
        cwd=str(ROOT),
        check=False,
    )
    if r.returncode != 0:
        raise SystemExit("Batch_01 dataset verification failed; refusing to train.")


def archive_previous() -> Path | None:
    if not PRIOR_BEST.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = ARCHIVES / f"tooth_detector_batch01_{stamp}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(OUT, dest)
    return dest


def main() -> None:
    force = "--force-retrain-batch01" in sys.argv
    if PRIOR_BEST.exists() and not force:
        raise SystemExit(
            "Refusing to overwrite models/detection/tooth_detector_batch01/.\n"
            "Train a new detector with:\n"
            "  python tools/train_tooth_detector_new_batch.py --batch 02\n"
            "To retrain Batch 01 anyway (archives the current run first):\n"
            "  python tools/train_tooth_detector_batch01.py --force-retrain-batch01"
        )
    verify()
    archived = archive_previous()
    # Prior run already early-stopped (patience=20) on this same split.
    # Continuing from that best.pt is not appropriate. Start from YOLO11n.
    init = ROOT / "models" / "detection" / "pretrained" / "yolo11n.pt"
    model = YOLO(str(init) if init.exists() else "yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        cache=True,
        save=True,
        exist_ok=True,
        project=str(ROOT / "models" / "detection"),
        name="tooth_detector_batch01",
        device="cpu",
        workers=0,
        plots=True,
        pretrained=True,
        val=True,
        seed=42,
        single_cls=True,
        verbose=True,
    )
    print("archived_previous", archived)
    print("best", OUT / "weights" / "best.pt")
    print("last", OUT / "weights" / "last.pt")


if __name__ == "__main__":
    main()
