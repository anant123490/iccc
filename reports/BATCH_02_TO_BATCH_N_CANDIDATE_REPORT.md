# Batch_02–N candidate tooth detections

YOLO11n (`models/tooth_detector_batch01/weights/best.pt`) was run on every JPG in `fdi_detection_dataset/images/selected/` **except** Batch_01 (`annotation_batches/Batch_01/seed_60`).

These are **candidate boxes only**. They are not QC-approved labels. No FDI. No ICDAS. Originals were not modified. No crops were written.

Predictions:

- labels: `predictions/labels/`
- overlays: `predictions/visualizations/`
- tables: `predictions/confidence_reports/`

## Counts

| Item | Value |
| --- | ---: |
| Images processed | 360 |
| Batch_01 filenames excluded | 60 |
| Total detections (conf ≥ 0.25) | 5493 |
| Average detections per image | 15.2583 |
| Mean confidence | 0.5447 |
| Zero teeth detected | 0 |
| Fewer than 8 teeth | 0 |
| More than 24 teeth | 2 |

Per-image CSV: `C:/Users/anant/OneDrive/Desktop/icdas project/predictions/confidence_reports/per_image.csv`.

## Confidence histogram (width 0.1)

| Bin (conf) | Count |
| --- | ---: |
| 0.2–0.3 | 497 |
| 0.3–0.4 | 969 |
| 0.4–0.5 | 1168 |
| 0.5–0.6 | 860 |
| 0.6–0.7 | 652 |
| 0.7–0.8 | 610 |
| 0.8–0.9 | 538 |
| 0.9–1.0 | 199 |

### Images with zero teeth detected

None.

### Images with fewer than 8 teeth

None.

### Images with more than 24 teeth (2)

```
anonymous_003-007-1215-01_1732863751319_Left_Lateral_View.jpg
anonymous_003-008-647-01_1729163338710_Mandibular_View.jpg
```

## Next (not done here)

Human QC of candidates into later batches. Do not treat these boxes as ground truth.
