# Batch_01 YOLO11n tooth detector results

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
| Precision | 0.7340 |
| Recall | 0.7500 |
| F1 | 0.7419 |
| mAP50 | 0.7643 |
| mAP50-95 | 0.2842 |

Best `results.csv` row by mAP50-95: epoch `24`. Training stopped early at epoch 44 (`patience=20`); best checkpoint is epoch 24.

## Fresh `model.val()` (best.pt)

### Val split

| Metric | Value |
| --- | ---: |
| Precision | 0.7351 |
| Recall | 0.7500 |
| F1 | 0.7425 |
| mAP50 | 0.7642 |
| mAP50-95 | 0.2840 |

### Test split

| Metric | Value |
| --- | ---: |
| Precision | 0.6737 |
| Recall | 0.7667 |
| F1 | 0.7172 |
| mAP50 | 0.7422 |
| mAP50-95 | 0.2988 |

Holdouts are tiny (6 val / 8 test images). Treat numbers as **seed-run estimates**, not production detector quality. A first AdamW run at `lr0=0.01` was archived under `models/tooth_detector_batch01_run1_adamw_lr0.01_collapsed/` and was not used.

## Loss curves

Copied from the Ultralytics run directory when present:

- `reports/batch01_yolo_plots/results.png`
- `reports/batch01_yolo_plots/confusion_matrix.png`
- `reports/batch01_yolo_plots/confusion_matrix_normalized.png`
- `reports/batch01_yolo_plots/BoxPR_curve.png`
- `reports/batch01_yolo_plots/BoxF1_curve.png`
- `reports/batch01_yolo_plots/BoxP_curve.png`
- `reports/batch01_yolo_plots/BoxR_curve.png`
- `reports/batch01_yolo_plots/val_batch0_pred.jpg`
- `reports/batch01_yolo_plots/val_batch0_labels.jpg`
- `reports/batch01_yolo_plots/labels.jpg`

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
