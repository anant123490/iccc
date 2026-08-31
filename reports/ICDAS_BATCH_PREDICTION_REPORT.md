# Stage 5B — ICDAS batch prediction

MobileNetV3 + CBAM (`deploy.keras`) ran on Stage 5A tooth crops only.

Did **not** retrain. Did **not** modify `dataset/`, `cropped_teeth/images/`, or YOLO weights. No Grad-CAM. No FDI. FastAPI/Streamlit were not wired.

## Inputs

| Item | Value |
| --- | --- |
| Crops | `cropped_teeth/images/` |
| Crops on disk | 5676 |
| Predicted | 5676 |
| Failed to read/preprocess | 0 |
| Model | `C:\Users\anant\OneDrive\Desktop\icdas project\models\deploy.keras` |
| Preprocess | PIL resize 224×224, BGR→RGB, float32 [0, 255] (same as training; ROI/CLAHE/specular/color-norm off) |

## Class distribution (predicted ICDAS 0–4)

| Class | Name | Count | Percent |
| --- | --- | ---: | ---: |
| 0 | ICDAS 0 | 2656 | 46.79% |
| 1 | ICDAS 1 | 0 | 0.00% |
| 2 | ICDAS 2 | 0 | 0.00% |
| 3 | ICDAS 3 | 0 | 0.00% |
| 4 | ICDAS 4 | 3020 | 53.21% |

## Confidence (argmax class probability)

| Stat | Value |
| --- | ---: |
| mean | 0.5076 |
| median | 0.5073 |
| min | 0.4119 |
| max | 0.5556 |
| stdev | 0.0115 |

## Confidence histogram

| Bin | Count |
| --- | ---: |
| 0.0–0.1 | 0 |
| 0.1–0.2 | 0 |
| 0.2–0.3 | 0 |
| 0.3–0.4 | 0 |
| 0.4–0.5 | 625 |
| 0.5–0.6 | 5051 |
| 0.6–0.7 | 0 |
| 0.7–0.8 | 0 |
| 0.8–0.9 | 0 |
| 0.9–1.0 | 0 |

## Files

- `predictions/icdas_predictions/predictions.csv`
- `predictions/icdas_predictions/class_counts.csv`
- `predictions/icdas_predictions/confidence_histogram.csv`

These grades are **model guesses on detector crops**, not verified ICDAS labels.

**Superseded as a clinical result.** See `reports/ICDAS_CLASSIFIER_AUDIT.md`: `deploy.keras` is a non-monotonic 4-output ordinal head; classes 1–3 are clipped away by decode. Do not use this CSV as ICDAS ground truth.

## Reuse (later FastAPI / Streamlit)

```python
from ml.src.icdas_predictor import IcdasCropClassifier

clf = IcdasCropClassifier()  # models/deploy.keras if present, else models/best.keras
pred = clf.predict_bgr(crop_bgr, crop_name="upload.jpg")
# pred.predicted_class, pred.confidence, pred.prob_0 … pred.prob_4
```

CLI: `python tools/run_icdas_batch_prediction.py`
