#!/usr/bin/env python3
"""Stage 5B: ICDAS 0–4 batch prediction on Stage 5A tooth crops.

Does not retrain, does not modify dataset/ or cropped_teeth/images/, no Grad-CAM, no FDI.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.src.icdas_predictor import (  # noqa: E402
    CLASS_NAMES,
    IcdasCropClassifier,
    IMAGE_SIZE,
)
from ml.src.tooth_cropping import read_bgr  # noqa: E402

CROPS = ROOT / "data" / "tooth_crops" / "generated" / "images"
MANIFEST = ROOT / "data" / "tooth_crops" / "generated" / "manifest.csv"
OUT = ROOT / "predictions" / "icdas_predictions"
REPORT = ROOT / "reports" / "ICDAS_BATCH_PREDICTION_REPORT.md"
BATCH = 32


def load_source_map() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return {r["crop_name"]: r.get("image_name", "") for r in csv.DictReader(f)}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    crops = sorted(CROPS.glob("*.jpg"))
    if not crops:
        raise SystemExit(f"no crops in {CROPS}")
    clf = IcdasCropClassifier()
    src_map = load_source_map()

    processed: list[np.ndarray] = []
    names: list[str] = []
    skipped: list[str] = []
    rows: list[dict] = []

    def flush() -> None:
        if not processed:
            return
        batch = np.stack(processed, axis=0)
        probs = clf.predict_processed_batch(batch)
        for name, p in zip(names, probs):
            pred = clf._pack(name, p)
            rows.append(
                {
                    "crop_name": pred.crop_name,
                    "image_name": src_map.get(pred.crop_name, ""),
                    "predicted_class": pred.predicted_class,
                    "class_name": pred.class_name,
                    "confidence": pred.confidence,
                    "prob_0": pred.prob_0,
                    "prob_1": pred.prob_1,
                    "prob_2": pred.prob_2,
                    "prob_3": pred.prob_3,
                    "prob_4": pred.prob_4,
                }
            )
        processed.clear()
        names.clear()

    for i, path in enumerate(crops, start=1):
        im = read_bgr(path)
        if im is None:
            skipped.append(path.name)
            continue
        try:
            x = clf.preprocess_bgr(im)
        except Exception:
            skipped.append(path.name)
            continue
        if x.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            skipped.append(path.name)
            continue
        processed.append(x)
        names.append(path.name)
        if len(processed) >= BATCH:
            flush()
        if i % 500 == 0:
            print(f"{i}/{len(crops)}", flush=True)
    flush()

    pred_fields = [
        "crop_name",
        "image_name",
        "predicted_class",
        "class_name",
        "confidence",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "prob_4",
    ]
    write_csv(OUT / "predictions.csv", rows, pred_fields)

    counts = Counter(int(r["predicted_class"]) for r in rows)
    n = len(rows)
    class_rows = []
    for g in range(5):
        c = counts.get(g, 0)
        class_rows.append(
            {
                "icdas_class": g,
                "class_name": CLASS_NAMES[g],
                "count": c,
                "percent": round(100.0 * c / n, 4) if n else 0.0,
            }
        )
    write_csv(
        OUT / "class_counts.csv",
        class_rows,
        ["icdas_class", "class_name", "count", "percent"],
    )

    confs = [float(r["confidence"]) for r in rows]
    hist_rows = []
    for b in range(10):
        lo = b / 10.0
        hi = (b + 1) / 10.0
        if b < 9:
            c = sum(1 for x in confs if lo <= x < hi)
        else:
            c = sum(1 for x in confs if lo <= x <= 1.0)
        hist_rows.append({"bin_start": f"{lo:.1f}", "bin_end": f"{hi:.1f}", "count": c})
    write_csv(
        OUT / "confidence_histogram.csv",
        hist_rows,
        ["bin_start", "bin_end", "count"],
    )

    summary = {
        "model": str(clf.model_path),
        "crops_dir": str(CROPS),
        "crops_on_disk": len(crops),
        "predictions": n,
        "unreadable_or_failed": skipped,
        "class_counts": {str(r["icdas_class"]): r["count"] for r in class_rows},
        "confidence_mean": statistics.mean(confs) if confs else 0,
        "confidence_median": statistics.median(confs) if confs else 0,
        "confidence_min": min(confs) if confs else 0,
        "confidence_max": max(confs) if confs else 0,
        "confidence_stdev": statistics.pstdev(confs) if len(confs) > 1 else 0,
        "preprocess": "training contract: PIL 224x224, BGR→RGB, float32 [0,255], no ROI/CLAHE/specular/color-norm",
        "head": "4-threshold ordinal decoded to ICDAS 0–4" if clf.ordinal else "5-class softmax",
        "note": "Auto-labels only. Not dentist ICDAS ground truth. No FDI. No Grad-CAM.",
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dist_md = "\n".join(
        f"| {r['icdas_class']} | {r['class_name']} | {r['count']} | {r['percent']:.2f}% |"
        for r in class_rows
    )
    hist_md = "\n".join(f"| {r['bin_start']}–{r['bin_end']} | {r['count']} |" for r in hist_rows)
    skip_n = len(skipped)
    md = f"""# Stage 5B — ICDAS batch prediction

MobileNetV3 + CBAM (`{clf.model_path.name}`) ran on Stage 5A tooth crops only.

Did **not** retrain. Did **not** modify `dataset/`, `cropped_teeth/images/`, or YOLO weights. No Grad-CAM. No FDI. FastAPI/Streamlit were not wired.

## Inputs

| Item | Value |
| --- | --- |
| Crops | `cropped_teeth/images/` |
| Crops on disk | {len(crops)} |
| Predicted | {n} |
| Failed to read/preprocess | {skip_n} |
| Model | `{clf.model_path}` |
| Preprocess | PIL resize 224×224, BGR→RGB, float32 [0, 255] (same as training; ROI/CLAHE/specular/color-norm off) |
| Head | {"4-threshold CORAL ordinal → 5-class ICDAS 0–4" if clf.ordinal else "5-class softmax"} |

## Class distribution (predicted ICDAS 0–4)

| Class | Name | Count | Percent |
| --- | --- | ---: | ---: |
{dist_md}

## Confidence (argmax class probability)

| Stat | Value |
| --- | ---: |
| mean | {summary["confidence_mean"]:.4f} |
| median | {summary["confidence_median"]:.4f} |
| min | {summary["confidence_min"]:.4f} |
| max | {summary["confidence_max"]:.4f} |
| stdev | {summary["confidence_stdev"]:.4f} |

## Confidence histogram

| Bin | Count |
| --- | ---: |
{hist_md}

## Files

- `predictions/icdas_predictions/predictions.csv`
- `predictions/icdas_predictions/class_counts.csv`
- `predictions/icdas_predictions/confidence_histogram.csv`

These grades are **model guesses on detector crops**, not verified ICDAS labels.

## Reuse (later FastAPI / Streamlit)

```python
from ml.src.icdas_predictor import IcdasCropClassifier

clf = IcdasCropClassifier()  # models/icdas/current/deploy.keras only; no stale ordinal
pred = clf.predict_bgr(crop_bgr, crop_name="upload.jpg")
# pred.predicted_class, pred.confidence, pred.prob_0 … pred.prob_4
```

CLI: `python tools/run_icdas_batch_prediction.py`
"""
    REPORT.write_text(md, encoding="utf-8")
    print(json.dumps({"predictions": n, "class_counts": summary["class_counts"], "model": str(clf.model_path)}, indent=2))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
