#!/usr/bin/env python3
"""Run Stage 5A tooth cropping on RGB intraoral images using Batch_01 best.pt.

Does not retrain YOLO, does not use Batch_02 labels, does not write into dataset/.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.src.tooth_cropping import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_SOURCE,
    DEFAULT_WEIGHTS,
    ToothCropPipeline,
    manifest_rows,
    skipped_rows,
)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    src = DEFAULT_SOURCE
    out = DEFAULT_OUT
    weights = DEFAULT_WEIGHTS
    images = sorted(src.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no JPG in {src}")

    pipe = ToothCropPipeline(weights=weights)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    all_manifest: list[dict] = []
    all_skip: list[dict] = []
    per_image_kept: list[int] = []
    per_image_raw: list[int] = []
    confs: list[float] = []
    skip_reasons: Counter[str] = Counter()

    for i, path in enumerate(images, start=1):
        result = pipe.crop_path(path)
        pipe.save_result(result, out)
        kept = manifest_rows(result)
        skipped = skipped_rows(result)
        all_manifest.extend(kept)
        all_skip.extend(skipped)
        per_image_kept.append(result.n_kept)
        per_image_raw.append(result.n_raw)
        confs.extend(r["confidence"] for r in kept)
        skip_reasons.update(s["skip_reason"] for s in skipped)
        if i % 40 == 0 or i == len(images):
            print(f"{i}/{len(images)} kept_crops={len(all_manifest)}", flush=True)

    write_csv(
        out / "manifest.csv",
        all_manifest,
        ["image_name", "crop_name", "confidence", "x1", "y1", "x2", "y2", "crop_w", "crop_h"],
    )
    write_csv(
        out / "skipped.csv",
        all_skip,
        ["image_name", "confidence", "x1", "y1", "x2", "y2", "skip_reason"],
    )

    n_img = len(images)
    n_crops = len(all_manifest)
    zero = sum(1 for n in per_image_kept if n == 0)
    summary = {
        "weights": str(weights),
        "source": str(src),
        "images_processed": n_img,
        "total_tooth_crops": n_crops,
        "raw_detections_before_qc": int(sum(per_image_raw)),
        "skipped_invalid_or_tiny": len(all_skip),
        "skip_reason_counts": dict(skip_reasons),
        "average_teeth_per_image": (n_crops / n_img) if n_img else 0,
        "min_teeth_detected": min(per_image_kept) if per_image_kept else 0,
        "max_teeth_detected": max(per_image_kept) if per_image_kept else 0,
        "images_with_zero_kept_crops": zero,
        "confidence_mean": statistics.mean(confs) if confs else 0,
        "confidence_median": statistics.median(confs) if confs else 0,
        "confidence_min": min(confs) if confs else 0,
        "confidence_max": max(confs) if confs else 0,
        "confidence_stdev": statistics.pstdev(confs) if len(confs) > 1 else 0,
        "note": "Crops are detector outputs, not ICDAS labels and not FDI numbers. dataset/ was not modified.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = ROOT / "reports" / "TOOTH_CROPPING_REPORT.md"
    skip_table = "\n".join(f"| `{k}` | {v} |" for k, v in sorted(skip_reasons.items())) or "| (none) | 0 |"
    md = f"""# Stage 5A — automatic tooth cropping

YOLO11n `best.pt` (Batch 01 human boxes only) ran on RGB intraoral photos in `fdi_detection_dataset/images/selected/`.

Originals were **not** modified. `dataset/` (ICDAS) was **not** modified. Batch 02 candidate labels were **not** used. The detector was **not** retrained.

## Outputs

- crops: `cropped_teeth/images/`
- overlays (green = kept, red = skipped QC): `cropped_teeth/overlays/`
- manifest: `cropped_teeth/manifest.csv`
- skipped QC log: `cropped_teeth/skipped.csv`

## Counts

| Item | Value |
| --- | ---: |
| Images processed | {summary["images_processed"]} |
| Raw detections (class tooth, conf ≥ 0.25) | {summary["raw_detections_before_qc"]} |
| Tooth crops written | {summary["total_tooth_crops"]} |
| Skipped (tiny/invalid/blank/extreme) | {summary["skipped_invalid_or_tiny"]} |
| Average teeth per image (kept) | {summary["average_teeth_per_image"]:.4f} |
| Min teeth / image | {summary["min_teeth_detected"]} |
| Max teeth / image | {summary["max_teeth_detected"]} |
| Images with zero kept crops | {summary["images_with_zero_kept_crops"]} |

## Confidence (kept crops)

| Stat | Value |
| --- | ---: |
| mean | {summary["confidence_mean"]:.4f} |
| median | {summary["confidence_median"]:.4f} |
| min | {summary["confidence_min"]:.4f} |
| max | {summary["confidence_max"]:.4f} |
| stdev | {summary["confidence_stdev"]:.4f} |

## QC skip reasons

| Reason | Count |
| --- | ---: |
{skip_table}

QC: min side {20} px, aspect ≤ 8, area 0.04%–75% of image, non-blank pixels, class `tooth` only, 8% box padding.

## Reuse (FastAPI / Streamlit)

```python
from ml.src.tooth_cropping import ToothCropPipeline

pipe = ToothCropPipeline()  # loads models/tooth_detector_batch01/weights/best.pt
result = pipe.crop_bgr(bgr_uint8, source_name="upload.jpg")
# result.crops_bgr: list[(crop_name, crop_bgr)]
# result.overlay_bgr: all boxes
# result.items: confidence + xyxy
```

CLI: `python tools/run_tooth_cropping.py`

Crops are **not** ICDAS grades. Do not copy them into `dataset/` until a dentist labels them 0–4.
"""
    report.write_text(md, encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("images_processed", "total_tooth_crops", "average_teeth_per_image")}, indent=2))
    print("wrote", report)


if __name__ == "__main__":
    main()
