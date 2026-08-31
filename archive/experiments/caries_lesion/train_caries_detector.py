"""Train YOLO caries-region detector. Classes D/d only — not ICDAS."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ultralytics.models import YOLO

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data_external" / "detection" / "public_caries" / "data.yaml"
OUT = ROOT / "models" / "caries_detector"
CLASSES = {0: "D", 1: "d"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classes.yaml").write_text(
        "nc: 2\nnames:\n  0: D\n  1: d\n"
        "# D = permanent-tooth decay region\n"
        "# d = primary-tooth decay region\n"
        "# NOT ICDAS grades\n",
        encoding="utf-8",
    )
    config = {
        "model": "yolov8n",
        "data": str(DATA),
        "epochs": 8,
        "imgsz": 320,
        "batch": 4,
        "device": "cpu",
        "workers": 0,
        "seed": 42,
        "classes": CLASSES,
        "note": "COCO-pretrained yolov8n backbone if present; not a tooth/FDI model. Labels are d/D decay regions.",
    }
    (OUT / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    t0 = time.time()
    weights_guess = ROOT / "yolov8n.pt"
    try:
        model = YOLO(str(weights_guess) if weights_guess.exists() else "yolov8n.pt")
    except Exception:
        model = YOLO("yolov8n.yaml")

    results = model.train(
        data=str(DATA),
        epochs=8,
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        project=str(OUT),
        name="train",
        exist_ok=True,
        seed=42,
        patience=5,
        plots=True,
        pretrained=True,
    )
    duration = time.time() - t0

    best = OUT / "train" / "weights" / "best.pt"
    last = OUT / "train" / "weights" / "last.pt"
    if best.exists():
        shutil.copy2(best, OUT / "best.pt")
    if last.exists():
        shutil.copy2(last, OUT / "last.pt")

    eval_model = YOLO(str(OUT / "best.pt")) if (OUT / "best.pt").exists() else model
    val_metrics = eval_model.val(data=str(DATA), split="val", imgsz=320, device="cpu", workers=0)
    test_metrics = eval_model.val(
        data=str(DATA),
        split="test",
        imgsz=320,
        device="cpu",
        workers=0,
        project=str(OUT),
        name="test_eval",
        exist_ok=True,
    )

    def pack(m) -> dict:
        box = getattr(m, "box", m)
        def g(name, default=None):
            v = getattr(box, name, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        return {
            "precision": g("mp"),
            "recall": g("mr"),
            "mAP50": g("map50"),
            "mAP50-95": g("map"),
        }

    metrics = {
        "training_seconds": round(duration, 1),
        "val": pack(val_metrics),
        "test": pack(test_metrics),
        "test_images": 277,
        "classes": CLASSES,
        "disclaimer": "Not clinical accuracy. Detection classes are d/D decay regions, not ICDAS.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
