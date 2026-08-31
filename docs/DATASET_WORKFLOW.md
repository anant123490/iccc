# Dataset workflows

Two independent ML tracks. Do not mix full-mouth photos with ICDAS class folders.

## 1. Tooth detection

```text
NEW FULL-MOUTH RGB IMAGE
+ RECTANGULAR TOOTH BOXES
    → data/detection/raw_images/ + data/detection/annotations/
    → data/detection/batches/batchNN/  (YOLO split)
    → python tools/train_tooth_detector_new_batch.py --batch NN
    → models/detection/tooth_detector_batchNN/
```

**Batch 01 already exists.** Canonical files:

- Images: `fdi_detection_dataset/images/selected/` (420)
- Split + labels: `fdi_detection_dataset/tooth_detector_batch01/` (46/6/8, 767 boxes)
- Weights: `models/detection/tooth_detector_batch01/weights/best.pt`

That tree was **not** renamed so Ultralytics `data.yaml`, crop defaults, and `annotation_batches` JSON keep working.

New photographs go to `data/detection/raw_images/`, not into Batch 01.

Details: `data/detection/README.md`.

Crops from the current detector:

```text
python tools/run_tooth_cropping.py
→ data/tooth_crops/generated/     (~5,676 files; not ICDAS GT)
```

## 2. ICDAS classification

```text
INDIVIDUAL TOOTH IMAGE
+ CLINICIAN-CONFIRMED ICDAS 0–4
    → data/icdas/raw/ then train|val|test/0–4/
    → quality check (tools/check_dataset.py)
    → python ml/train.py --config ml/configs/default.yaml
    → MobileNetV3 + CBAM, 5-class softmax
    → evaluate
    → only then models/icdas/current/deploy.keras
```

**Blocked today:** `annotations/annotations.csv` has ~643 rows; class folders have **no pixels**. Do not train. Do not auto-label detector crops. Do not replace historical `deploy.keras`.

ICDAS 5 and 6 stay in `data/icdas/excluded/`. Never fold them into class 4.

Details: `data/icdas/README.md`.
