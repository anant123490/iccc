#!/usr/bin/env python3
"""Write reports/BATCH_02_TO_BATCH_N_CANDIDATE_REPORT.md from prediction summary."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "predictions" / "confidence_reports" / "summary.json"
CSV = ROOT / "predictions" / "confidence_reports" / "per_image.csv"
OUT = ROOT / "reports" / "BATCH_02_TO_BATCH_N_CANDIDATE_REPORT.md"


def _list_block(title: str, names: list[str]) -> str:
    if not names:
        return f"### {title}\n\nNone.\n"
    lines = [f"### {title} ({len(names)})", "", "```"]
    lines.extend(names)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"missing {SUMMARY}; run tools/predict_tooth_detector_remaining.py first")
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    hist = data.get("confidence_histogram_0.1") or {}
    hist_rows = "\n".join(f"| {k}–{float(k)+0.1:.1f} | {v} |" for k, v in hist.items()) or "| (empty) | 0 |"

    md = f"""# Batch_02–N candidate tooth detections

YOLO11n (`models/detection/tooth_detector_batch01/weights/best.pt`) was run on every JPG in `fdi_detection_dataset/images/selected/` **except** Batch_01 (`annotation_batches/Batch_01/seed_60`).

These are **candidate boxes only**. They are not QC-approved labels. No FDI. No ICDAS. Originals were not modified. No crops were written.

Predictions:

- labels: `predictions/labels/`
- overlays: `predictions/visualizations/`
- tables: `predictions/confidence_reports/`

## Counts

| Item | Value |
| --- | ---: |
| Images processed | {data.get("images_processed")} |
| Batch_01 filenames excluded | {data.get("batch01_excluded")} |
| Total detections (conf ≥ 0.25) | {data.get("total_detections")} |
| Average detections per image | {_fmt(data.get("average_detections_per_image"))} |
| Mean confidence | {_fmt(data.get("mean_confidence"))} |
| Zero teeth detected | {len(data.get("zero_teeth") or [])} |
| Fewer than 8 teeth | {len(data.get("fewer_than_8") or [])} |
| More than 24 teeth | {len(data.get("more_than_24") or [])} |

Per-image CSV: `{CSV.as_posix()}`.

## Confidence histogram (width 0.1)

| Bin (conf) | Count |
| --- | ---: |
{hist_rows}

{_list_block("Images with zero teeth detected", list(data.get("zero_teeth") or []))}
{_list_block("Images with fewer than 8 teeth", list(data.get("fewer_than_8") or []))}
{_list_block("Images with more than 24 teeth", list(data.get("more_than_24") or []))}
## Next (not done here)

Human QC of candidates into later batches. Do not treat these boxes as ground truth.
"""
    OUT.write_text(md, encoding="utf-8")
    print("wrote", OUT)


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    main()
