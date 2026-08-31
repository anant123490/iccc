#!/usr/bin/env python3
"""Write reports/BATCH_01_YOLO_RESULTS.md from Ultralytics run artifacts."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "models" / "detection" / "tooth_detector_batch01"
DATA = ROOT / "fdi_detection_dataset" / "tooth_detector_batch01" / "data.yaml"
WEIGHTS = RUN / "weights" / "best.pt"
OUT = ROOT / "reports" / "BATCH_01_YOLO_RESULTS.md"
PLOT_DIR = ROOT / "reports" / "batch01_yolo_plots"


def _best_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit(f"empty results csv {csv_path}")
    key = "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in rows[0] else "metrics/mAP50-95"
    return max(rows, key=lambda r: float(r.get(key) or 0))


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _copy_plots() -> list[str]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    names = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
        "labels.jpg",
    ]
    copied = []
    for name in names:
        src = RUN / name
        if src.exists():
            dst = PLOT_DIR / name
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def _fmt(v: str | float) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"missing {WEIGHTS}")
    results_csv = RUN / "results.csv"
    if not results_csv.exists():
        raise SystemExit(f"missing {results_csv}")

    last = _best_row(results_csv)
    p = float(last.get("metrics/precision(B)", last.get("metrics/precision", 0)))
    r = float(last.get("metrics/recall(B)", last.get("metrics/recall", 0)))
    map50 = float(last.get("metrics/mAP50(B)", last.get("metrics/mAP50", 0)))
    map5095 = float(last.get("metrics/mAP50-95(B)", last.get("metrics/mAP50-95", 0)))
    f1 = _f1(p, r)

    model = YOLO(str(WEIGHTS))
    val_metrics = model.val(data=str(DATA), split="val", imgsz=640, device="cpu", plots=True, verbose=False)
    test_metrics = model.val(data=str(DATA), split="test", imgsz=640, device="cpu", plots=True, verbose=False)

    copied = _copy_plots()

    md = f"""# Batch_01 YOLO11n tooth detector results

Single-class detector: `0 = tooth`. Trained only on Batch_01 (60 human-QC images, 767 boxes).

This is **not** FDI numbering and **not** ICDAS grading.

## Dataset

| Split | Images | Path |
| --- | ---: | --- |
| train | 46 | `fdi_detection_dataset/tooth_detector_batch01/images/train` |
| val | 6 | `fdi_detection_dataset/tooth_detector_batch01/images/val` |
| test | 8 | `fdi_detection_dataset/tooth_detector_batch01/images/test` |

Split is patient-id aware where possible (seed 42). Pairing is 1:1 JPG/TXT.

## Training setup

| Item | Value |
| --- | --- |
| model | YOLO11n (`yolo11n.pt` COCO init, `nc=1`) |
| imgsz | 640 |
| epochs (requested) | 100 |
| patience | 20 |
| batch | 8 (CPU; 16 requested) |
| optimizer | AdamW (`lr0=0.001`; default 0.01 collapsed a prior archived run) |
| device | CPU (`torch 2.2.2+cpu`) |
| cache | True |
| weights | `models/tooth_detector_batch01/weights/best.pt` |

`save_best_only` is not an Ultralytics train flag. Ultralytics still writes `best.pt` (best fitness) and `last.pt`.

## Validation metrics (last epoch in `results.csv`)

| Metric | Value |
| --- | ---: |
| Precision | {_fmt(p)} |
| Recall | {_fmt(r)} |
| F1 | {_fmt(f1)} |
| mAP50 | {_fmt(map50)} |
| mAP50-95 | {_fmt(map5095)} |

Best `results.csv` row by mAP50-95: epoch `{last.get("epoch", "?")}`. Training stopped early at epoch 44 (`patience=20`); best checkpoint is epoch 24.

## Fresh `model.val()` (best.pt)

### Val split

| Metric | Value |
| --- | ---: |
| Precision | {_fmt(float(val_metrics.box.mp))} |
| Recall | {_fmt(float(val_metrics.box.mr))} |
| F1 | {_fmt(_f1(float(val_metrics.box.mp), float(val_metrics.box.mr)))} |
| mAP50 | {_fmt(float(val_metrics.box.map50))} |
| mAP50-95 | {_fmt(float(val_metrics.box.map))} |

### Test split

| Metric | Value |
| --- | ---: |
| Precision | {_fmt(float(test_metrics.box.mp))} |
| Recall | {_fmt(float(test_metrics.box.mr))} |
| F1 | {_fmt(_f1(float(test_metrics.box.mp), float(test_metrics.box.mr)))} |
| mAP50 | {_fmt(float(test_metrics.box.map50))} |
| mAP50-95 | {_fmt(float(test_metrics.box.map))} |

Holdouts are tiny (6 val / 8 test images). Treat numbers as **seed-run estimates**, not production detector quality. A first AdamW run at `lr0=0.01` was archived under `models/tooth_detector_batch01_run1_adamw_lr0.01_collapsed/` and was not used.

## Loss curves

Copied from the Ultralytics run directory when present:

"""
    for name in copied:
        md += f"- `reports/batch01_yolo_plots/{name}`\n"
    md += """
Primary curve plot: `reports/batch01_yolo_plots/results.png` (box / cls / dfl losses and metrics vs epoch).

## Confusion matrix

- `reports/batch01_yolo_plots/confusion_matrix.png`
- `reports/batch01_yolo_plots/confusion_matrix_normalized.png` (if generated)

Single-class detection confusion is tooth vs background.

## Sample predictions on validation images

Ultralytics writes overlay batches during training/val:

- `reports/batch01_yolo_plots/val_batch0_pred.jpg`
- `reports/batch01_yolo_plots/val_batch0_labels.jpg` (ground truth)

## Caveats

- 60 images only. Val/test are small.
- CPU training; batch 8 instead of 16.
- NumPy was pinned to 1.26.4 so PyTorch 2.2 CPU could call `torch.from_numpy`.
- Batch_01 seed copies live under `annotation_batches/Batch_01/seed_60` and `tooth_detector_batch01/`. `fdi_detection_dataset/images/selected/` was not modified.
"""
    OUT.write_text(md, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
