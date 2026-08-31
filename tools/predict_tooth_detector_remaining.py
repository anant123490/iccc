#!/usr/bin/env python3
"""Predict tooth boxes on selected/ excluding Batch_01. Does not modify originals."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "fdi_detection_dataset" / "images" / "selected"
SEED = ROOT / "annotation_batches" / "Batch_01" / "seed_60"
WEIGHTS = ROOT / "models" / "detection" / "tooth_detector_batch01" / "weights" / "best.pt"
PRED = ROOT / "predictions"


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"missing weights {WEIGHTS}")
    skip = {p.name for p in SEED.glob("*.jpg")}
    images = sorted(p for p in SELECTED.glob("*.jpg") if p.name not in skip)
    (PRED / "labels").mkdir(parents=True, exist_ok=True)
    (PRED / "visualizations").mkdir(parents=True, exist_ok=True)
    (PRED / "confidence_reports").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))
    rows = []
    confs = []
    zero, low, high = [], [], []
    for img_path in images:
        results = model.predict(source=str(img_path), imgsz=640, conf=0.25, device="cpu", verbose=False)
        r = results[0]
        raw = np.fromfile(str(img_path), dtype=np.uint8)
        im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        lines = []
        n = 0
        cs = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                c = float(b.conf[0])
                cls = int(b.cls[0])
                if cls != 0:
                    continue
                n += 1
                cs.append(c)
                confs.append(c)
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {c:.4f}")
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    im,
                    f"tooth {c:.2f}",
                    (int(x1), max(15, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        (PRED / "labels" / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        ok, enc = cv2.imencode(".jpg", im)
        enc.tofile(str(PRED / "visualizations" / img_path.name))
        mean_c = sum(cs) / len(cs) if cs else 0.0
        rows.append({"filename": img_path.name, "n_det": n, "mean_conf": round(mean_c, 4)})
        if n == 0:
            zero.append(img_path.name)
        if n < 8:
            low.append(img_path.name)
        if n > 24:
            high.append(img_path.name)

    with (PRED / "confidence_reports" / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=["filename", "n_det", "mean_conf"])
        wri.writeheader()
        wri.writerows(rows)

    hist = {}
    for c in confs:
        bucket = f"{int(c * 10) / 10:.1f}"
        hist[bucket] = hist.get(bucket, 0) + 1

    summary = {
        "images_processed": len(images),
        "batch01_excluded": len(skip),
        "zero_teeth": zero,
        "fewer_than_8": low,
        "more_than_24": high,
        "total_detections": len(confs),
        "average_detections_per_image": (len(confs) / len(images)) if images else 0,
        "mean_confidence": (sum(confs) / len(confs)) if confs else 0,
        "confidence_histogram_0.1": dict(sorted(hist.items())),
        "note": "CANDIDATE detections only. Not ground truth. No FDI. No ICDAS. Originals not modified.",
    }
    (PRED / "confidence_reports" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["images_processed", "total_detections", "average_detections_per_image"]}, indent=2))
    print("zero", len(zero), "lt8", len(low), "gt24", len(high))


if __name__ == "__main__":
    main()
