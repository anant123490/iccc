#!/usr/bin/env python3
"""Train Tooth Detector V2 on gold dataset. Does not touch Batch 01 weights or source data."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "detection" / "gold_detector_dataset" / "data.yaml"
OUT = ROOT / "models" / "detection" / "tooth_detector_v2"
B01_WEIGHTS = ROOT / "models" / "detection" / "tooth_detector_batch01" / "weights" / "best.pt"
VIS = ROOT / "reports" / "tooth_detector_v2_visual_test"
REAL = ROOT / "reports" / "tooth_detector_v2_real_world_test"
METRICS = OUT / "eval_metrics.json"
SEED = 42
IMGSZ = 640
DEVICE = "cpu"


def f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def pack(m) -> dict:
    p, r = float(m.box.mp), float(m.box.mr)
    return {
        "precision": p,
        "recall": r,
        "f1": f1(p, r),
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
    }


def pick_images(split: str, n: int) -> list[Path]:
    folder = ROOT / "data" / "detection" / "gold_detector_dataset" / split / "images"
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng = random.Random(SEED)
    if len(imgs) <= n:
        return imgs
    return rng.sample(imgs, n)


def overlays(model: YOLO, split: str, dest_name: str, n: int = 10) -> None:
    srcs = pick_images(split, n)
    tmp_dir = VIS / f"_src_{split}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in srcs:
        shutil.copy2(p, tmp_dir / p.name)
    model.predict(
        source=str(tmp_dir),
        imgsz=IMGSZ,
        device=DEVICE,
        conf=0.25,
        save=True,
        project=str(VIS),
        name=dest_name,
        exist_ok=True,
        verbose=False,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)


def write_real_world_readme() -> None:
    REAL.mkdir(parents=True, exist_ok=True)
    incoming = REAL / "incoming"
    incoming.mkdir(exist_ok=True)
    (incoming / ".gitkeep").write_text("", encoding="utf-8")
    (REAL / "README.md").write_text(
        """# Tooth Detector V2 — real-world inference (unseen photos)

This folder is for **new intraoral photos that were not used in Gold train/valid/test**.

Do **not** copy Gold dataset images here. Do **not** use Batch 01 training photos.

## What to put here

1. Copy **20–30 new** RGB intraoral photos into `incoming/` (JPG/PNG).
2. Use photos from the clinic camera if possible (native resolution is fine).

## Run inference (does not train, does not overwrite Batch 01)

From the repo root, using the same Python that has Ultralytics (Batch 01 used system Python 3.12):

```text
C:\\Users\\anant\\AppData\\Local\\Programs\\Python\\Python312\\python.exe -c "from ultralytics import YOLO; YOLO(r'models/detection/tooth_detector_v2/weights/best.pt').predict(source=r'reports/tooth_detector_v2_real_world_test/incoming', imgsz=640, conf=0.25, save=True, project=r'reports/tooth_detector_v2_real_world_test', name='predictions', exist_ok=True)"
```

Overlays will appear in `predictions/`.

Optional: also run Batch 01 `models/detection/tooth_detector_batch01/weights/best.pt` the same way into `predictions_batch01/` for a side-by-side check. Do not replace either weight file.
""",
        encoding="utf-8",
    )


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}")
    if B01_WEIGHTS.exists() and B01_WEIGHTS.resolve() == (OUT / "weights" / "best.pt"):
        raise SystemExit("refusing to overwrite Batch 01")
    write_real_world_readme()
    OUT.mkdir(parents=True, exist_ok=True)
    VIS.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=100,
        patience=20,
        batch=8,
        imgsz=IMGSZ,
        device=DEVICE,
        workers=0,
        project=str(ROOT / "models" / "detection"),
        name="tooth_detector_v2",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        seed=SEED,
        deterministic=True,
        single_cls=True,
        cache=True,
        plots=True,
        verbose=True,
    )

    best = OUT / "weights" / "best.pt"
    last = OUT / "weights" / "last.pt"
    if not best.exists() or not last.exists():
        raise SystemExit(f"missing weights under {OUT / 'weights'}")

    v2 = YOLO(str(best))
    val_m = v2.val(data=str(DATA), split="val", imgsz=IMGSZ, device=DEVICE, plots=True, verbose=False)
    test_m = v2.val(data=str(DATA), split="test", imgsz=IMGSZ, device=DEVICE, plots=False, verbose=False)

    b01_on_gold_val = b01_on_gold_test = None
    if B01_WEIGHTS.exists():
        b01 = YOLO(str(B01_WEIGHTS))
        b01_on_gold_val = pack(b01.val(data=str(DATA), split="val", imgsz=IMGSZ, device=DEVICE, plots=False, verbose=False))
        b01_on_gold_test = pack(b01.val(data=str(DATA), split="test", imgsz=IMGSZ, device=DEVICE, plots=False, verbose=False))

    overlays(v2, "valid", "val_overlays", 10)
    overlays(v2, "test", "test_overlays", 10)

    payload = {
        "dataset": str(DATA),
        "model": "yolo11n",
        "epochs": 100,
        "patience": 20,
        "imgsz": IMGSZ,
        "batch": 8,
        "device": DEVICE,
        "best": str(best),
        "last": str(last),
        "v2_val": pack(val_m),
        "v2_test": pack(test_m),
        "b01_on_gold_val": b01_on_gold_val,
        "b01_on_gold_test": b01_on_gold_test,
        "b01_original_val": {"precision": 0.7537, "recall": 0.7361, "f1": 0.7448, "mAP50": 0.7451, "mAP50-95": 0.2826},
        "b01_original_test": {"precision": 0.6999, "recall": 0.7255, "f1": 0.7124, "mAP50": 0.7181, "mAP50-95": 0.2815},
    }
    METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", METRICS)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
