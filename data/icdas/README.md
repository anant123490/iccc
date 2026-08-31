# ICDAS classification dataset

This folder is for **individual tooth images** with **clinician-confirmed ICDAS 0–4** labels.

It is **not** the full-mouth detection dataset (`data/detection/` / `fdi_detection_dataset/`).

Detector crops in `data/tooth_crops/generated/` are **not** ICDAS ground truth. Do not copy them here until a clinician confirms the grade.

**Do not train a new ICDAS model until labeled pixels exist in `train/val/test/0–4`.**
Do not auto-label the 5,676 generated crops.
Do not use historical stale ordinal keras files as the production classifier.

ICDAS **5** and **6** are out of scope. Never silently convert them to ICDAS 4. Keep them under `excluded/`.

## Current status (blocked)

| Item | Status |
|------|--------|
| `annotations/annotations.csv` | ~643 rows preserved (classes include 1:145, 2:118, 3:121) |
| `train/val/test/0–4/` pixel files | **Missing** (folders exist, images are not on disk) |
| Intended model | MobileNetV3-Small + CBAM, 5-class softmax, `ordinal_regression: false` |
| On-disk keras | No approved production classifier. Stale ordinal files are not used at runtime. |
| `models/icdas/current/` | Empty — no approved production classifier |

## 1. Where to put raw ICDAS images

Unsorted incoming tooth photographs (before QC):

```text
data/icdas/raw/
```

Optional holding area before split:

```text
data/icdas/images/
```

## 2. Where to put annotation / label information

Canonical table:

```text
data/icdas/annotations/annotations.csv
```

Suggested columns:

```csv
filename,icdas_score,split,patient_id,notes
train/1/tooth_001.jpg,1,train,P001,clinician confirmed
```

Labeling studio CSV (human UI): `data/icdas/annotations/labeling_studio/labels.csv`

v2 labeling pool: `data/icdas/labeling_v2/`

## 3. How ICDAS 0–4 labels are represented

Folder name **and** CSV `icdas_score` use integers:

| Folder / score | Meaning |
|----------------|---------|
| 0 | Sound |
| 1 | First visual change in enamel |
| 2 | Distinct visual change in enamel |
| 3 | Localized enamel breakdown |
| 4 | Underlying dark shadow / dentin involvement |

Training config: `num_classes: 5`, softmax, not ordinal.

## 4. How to handle class 0

Class 0 is a real class. Put sound-tooth crops in `train/0/`, `val/0/`, `test/0/`. Do not omit 0 from the softmax head.

## 5. How to handle classes 1–4

Same layout: `train/1/` … `train/4/`. Only **clinician-confirmed** grades. Model predictions and public-dataset region names are not ICDAS grades.

## 6. How to prevent duplicates across splits

- One image file → one split
- Split by **patient_id** or **source full-mouth image**, never by random crop if crops share a mouth
- After copying, run `python tools/check_dataset.py`

## 7. How to split the dataset

Typical: 70% train / 15% val / 15% test, fixed seed, patient-aware.

```text
python tools/build_dataset.py
python ml/scripts/sync_annotations.py
python ml/scripts/validate_dataset.py --allow-empty
```

`sync_annotations.py` rebuilds `annotations/annotations.csv` from folders **only when images exist**. It will not wipe the current 643-row CSV when folders are empty.

## 8. How to check class balance

```text
python tools/check_dataset.py
python tools/audit_icdas_classifier.py
```

## 9. How to train a new model

Only after pixels exist for classes you intend to train:

```text
python ml/train.py --config ml/configs/default.yaml
```

Checkpoints go under `models/icdas/current/<experiment_name>/`.
`overwrite_root_checkpoints` is **false** so `current/deploy.keras` is not written until you turn it on after evaluation.

## 10. How to evaluate

Metrics are written next to the experiment folder (`test_evaluation/`, confusion matrix, classification report).

```text
python tools/evaluate_icdas.py
```

## 11. When a model can become the new `deploy.keras`

A candidate may be copied to `models/icdas/current/deploy.keras` **only if**:

1. Head is **5-class softmax** (not 4-output ordinal)
2. Trained on clinician-confirmed ICDAS 0–4 pixels (not auto-labeled YOLO crops)
3. Held-out metrics are reviewed and documented in `reports/`
4. ICDAS 5/6 were not remapped into class 4
5. You explicitly set `overwrite_root_checkpoints: true` **or** copy the file by hand after review

The historical stale ordinal checkpoint is **not** a production classifier and must not be loaded by FastAPI. Place an approved 5-class softmax model at `models/icdas/current/deploy.keras` after evaluation.

## Excluded grades

`data/icdas/excluded/5/` and `excluded/6/` hold out-of-scope images. Do not train on them.
