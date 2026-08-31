#!/usr/bin/env python3
"""Concise training report for Batch_01 tooth detector. Test split only for final numbers."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "models" / "detection" / "tooth_detector_batch01"
DATA = ROOT / "fdi_detection_dataset" / "tooth_detector_batch01" / "data.yaml"
WEIGHTS = RUN / "weights" / "best.pt"
LAST = RUN / "weights" / "last.pt"
VERIFY = ROOT / "reports" / "batch01_dataset_verify.json"
OUT = ROOT / "reports" / "TOOTH_DETECTOR_BATCH01_TRAINING.md"
PLOT_DIR = ROOT / "reports" / "batch01_yolo_plots"


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _best_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    key = "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in rows[0] else "metrics/mAP50-95"
    return max(rows, key=lambda r: float(r.get(key) or 0))


def main() -> None:
    if not WEIGHTS.exists() or not LAST.exists():
        raise SystemExit("missing best.pt or last.pt")
    verify = json.loads(VERIFY.read_text(encoding="utf-8")) if VERIFY.exists() else {}
    last = _best_row(RUN / "results.csv")
    model = YOLO(str(WEIGHTS))
    val_m = model.val(data=str(DATA), split="val", imgsz=640, device="cpu", plots=True, verbose=False)
    test_m = model.val(data=str(DATA), split="test", imgsz=640, device="cpu", plots=True, verbose=False)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plots = []
    for name in (
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
    ):
        src = RUN / name
        if src.exists():
            shutil.copy2(src, PLOT_DIR / name)
            plots.append(name)

    vp, vr = float(val_m.box.mp), float(val_m.box.mr)
    tp, tr = float(test_m.box.mp), float(test_m.box.mr)
    splits = verify.get("splits") or {}

    md = f"""# Tooth detector training (Batch 01 only)

Human-verified Batch 01 boxes only. Class `0 = tooth`. **Not** FDI. **Not** ICDAS.

Did **not** train on `predictions/` (360 remaining images) or `annotation_batches/Batch_02/yolo_candidate_labels/`.
Did **not** modify `fdi_detection_dataset/images/selected/`, `dataset/`, or ICDAS models.

## Dataset verification

| Split | Images | Labels | Boxes |
| --- | ---: | ---: | ---: |
| train | {splits.get("train", {}).get("images", "?")} | {splits.get("train", {}).get("labels", "?")} | {splits.get("train", {}).get("boxes", "?")} |
| val | {splits.get("val", {}).get("images", "?")} | {splits.get("val", {}).get("labels", "?")} | {splits.get("val", {}).get("boxes", "?")} |
| test (untouched) | {splits.get("test", {}).get("images", "?")} | {splits.get("test", {}).get("labels", "?")} | {splits.get("test", {}).get("boxes", "?")} |

- `nc`: 1
- names: `0: tooth`
- class histogram: `{verify.get("class_histogram")}`
- total boxes: {verify.get("total_boxes")}
- pairing: 1:1 JPG/TXT per split (verified)

## Configuration

| Item | Value |
| --- | --- |
| init | YOLO11n (`yolo11n.pt`); prior Batch 01 `best.pt` was archived, not continued (that run had already early-stopped) |
| data | `fdi_detection_dataset/tooth_detector_batch01/data.yaml` |
| imgsz | 640 |
| epochs | 100 (patience 20) |
| batch | 8 (CPU) |
| optimizer | AdamW, `lr0=0.001` |
| device | CPU |
| seed | 42 |
| best | `models/detection/tooth_detector_batch01/weights/best.pt` |
| last | `models/detection/tooth_detector_batch01/weights/last.pt` |

## Val (`best.pt`)

| Metric | Value |
| --- | ---: |
| Precision | {_fmt(vp)} |
| Recall | {_fmt(vr)} |
| F1 | {_fmt(_f1(vp, vr))} |
| mAP50 | {_fmt(float(val_m.box.map50))} |
| mAP50-95 | {_fmt(float(val_m.box.map))} |

Best logged epoch by mAP50-95: `{last.get("epoch", "?")}`.

## Test (`best.pt`, held-out Batch 01 test)

| Metric | Value |
| --- | ---: |
| Precision | {_fmt(tp)} |
| Recall | {_fmt(tr)} |
| F1 | {_fmt(_f1(tp, tr))} |
| mAP50 | {_fmt(float(test_m.box.map50))} |
| mAP50-95 | {_fmt(float(test_m.box.map))} |

Test is 8 images / 90 boxes. Treat as a seed estimate.

## Curves and plots

"""
    for name in plots:
        md += f"- `reports/batch01_yolo_plots/{name}`\n"
    md += """
Training curves: `reports/batch01_yolo_plots/results.png`.
"""
    OUT.write_text(md, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
