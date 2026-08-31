# Tooth detector training (Batch 01 only)

Human-verified Batch 01 boxes only. Class `0 = tooth`. **Not** FDI. **Not** ICDAS.

Did **not** train on `predictions/` (360 remaining images) or `annotation_batches/Batch_02/yolo_candidate_labels/`.
Did **not** modify `fdi_detection_dataset/images/selected/`, `dataset/`, or ICDAS models.

## Dataset verification

| Split | Images | Labels | Boxes |
| --- | ---: | ---: | ---: |
| train | 46 | 46 | 605 |
| val | 6 | 6 | 72 |
| test (untouched) | 8 | 8 | 90 |

- `nc`: 1
- names: `0: tooth`
- class histogram: `{'0': 767}`
- total boxes: 767
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
| best | `models/tooth_detector_batch01/weights/best.pt` |
| last | `models/tooth_detector_batch01/weights/last.pt` |

## Val (`best.pt`)

| Metric | Value |
| --- | ---: |
| Precision | 0.7537 |
| Recall | 0.7361 |
| F1 | 0.7448 |
| mAP50 | 0.7451 |
| mAP50-95 | 0.2826 |

Best logged epoch by mAP50-95: `41`.

## Test (`best.pt`, held-out Batch 01 test)

| Metric | Value |
| --- | ---: |
| Precision | 0.6999 |
| Recall | 0.7255 |
| F1 | 0.7124 |
| mAP50 | 0.7181 |
| mAP50-95 | 0.2815 |

Test is 8 images / 90 boxes. Treat as a seed estimate.

## Curves and plots

- `reports/batch01_yolo_plots/results.png`
- `reports/batch01_yolo_plots/confusion_matrix.png`
- `reports/batch01_yolo_plots/confusion_matrix_normalized.png`
- `reports/batch01_yolo_plots/BoxPR_curve.png`
- `reports/batch01_yolo_plots/BoxF1_curve.png`
- `reports/batch01_yolo_plots/val_batch0_pred.jpg`
- `reports/batch01_yolo_plots/val_batch0_labels.jpg`

Training curves: `reports/batch01_yolo_plots/results.png`.
