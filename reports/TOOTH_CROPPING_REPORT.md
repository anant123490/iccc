# Stage 5A — automatic tooth cropping

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
| Images processed | 420 |
| Raw detections (class tooth, conf ≥ 0.25) | 5676 |
| Tooth crops written | 5676 |
| Skipped (tiny/invalid/blank/extreme) | 0 |
| Average teeth per image (kept) | 13.5143 |
| Min teeth / image | 6 |
| Max teeth / image | 27 |
| Images with zero kept crops | 0 |

## Confidence (kept crops)

| Stat | Value |
| --- | ---: |
| mean | 0.5558 |
| median | 0.5592 |
| min | 0.2502 |
| max | 0.9103 |
| stdev | 0.1709 |

## QC skip reasons

| Reason | Count |
| --- | ---: |
| (none) | 0 |

QC: min side 20 px, aspect ≤ 8, area 0.04%–75% of image, non-blank pixels, class `tooth` only, 8% box padding.

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
