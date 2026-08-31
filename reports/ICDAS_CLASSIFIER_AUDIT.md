# ICDAS classifier audit (Stage 5B follow-up)

Date: 2026-08-27  
JSON companion: `reports/icdas_classifier_audit.json`

This audit does **not** treat the 5,676 YOLO-crop predictions as labels. Detector was not retrained. `deploy.keras` / `best.keras` were not overwritten.

## A. FDI status

**FDI is out of active scope.** Confirmed in `docs/PROJECT_SCOPE.md`. No FDI labels, models, or numbering were added. Existing whole-tooth detection assets were preserved (the `fdi_detection_dataset/` directory name is historical).

Active pipeline:

Patient Registration → Visit → RGB photo → whole-tooth detection → crop → ICDAS 0–4 → Grad-CAM → report.

## B. Detection status

**Still usable.** YOLO11n `models/tooth_detector_batch01/weights/best.pt` (Batch 01 human boxes only). Held-out test: P 0.700, R 0.726, F1 0.712, mAP50 0.718, mAP50-95 0.282. Crops: **5,676** under `cropped_teeth/` from 420 RGB images. Originals were not modified.

## C. Why predictions are only ICDAS 0 and 4

Not because classes 1–3 were missing from the **CSV** (they are listed). Cause is the **head + decode**:

1. `models/deploy.keras` output layer name is **`ordinal`**, shape **`(None, 4)`**. That is CORAL-style `P(y > k)` for k = 0,1,2,3, **not** 5-class softmax.
2. `ValueError: expected 5-class softmax, got (None, 4)` is the production contract (`backend/app/inference.py`, `ml/configs/default.yaml`: `ordinal_regression: false`) meeting a **different** exported checkpoint.
3. Decoding 4 sigmoids to 5 classes uses `P(0)=1-P(y>0)`, `P(k)=P(y>k-1)-P(y>k)`, `P(4)=P(y>3)`, then clip negatives and renormalize (`ml/src/losses.py`).
4. On 64 YOLO crops, **raw means are non-monotonic** (they *increase* with k): about `[0.37, 0.46, 0.58, 0.64]`. CORAL requires **decreasing** `P(y>k)`. Every middle difference is negative → clipped to **0**. After renormalize, mass sits on **0 vs 4** only. Argmax confidence stays ~0.5. The 5,676-crop CSV (2656 / 0 / 0 / 0 / 3020) is this collapse, not a clinical 0/4 prevalence.

Silent ordinal→softmax conversion was **not** a valid clinical fix. `IcdasCropClassifier` can decode ordinal tensors; that does not make `deploy.keras` a validated 5-class ICDAS model.

## D. Model architecture

| Checkpoint | Input | Output | Layer name | Kind |
| --- | --- | --- | --- | --- |
| `models/deploy.keras` | `(None, 224, 224, 3)` | **`(None, 4)`** | `ordinal` | 4-threshold ordinal |
| `models/best.keras` | same | **`(None, 4)`** | `ordinal` | same |

Intended (`ml/configs/default.yaml`, `ml/src/model.py` default): **5-class softmax**, `ordinal_regression: false`, loss `sparse_categorical_crossentropy`.

Historical `models/icdas_mobilenet_cbam/config.json`: `num_classes: 7`, `ordinal_regression: true`, extra preprocess (ROI/CLAHE/specular/color-norm **on**). Current training/inference configs have those flags **off**. The root `deploy.keras` is **stale relative to the documented production contract** (ordinal vs softmax; possible older 7-class lineage vs 0–4).

ICDAS 5–6 remain out of scope and were not remapped.

## E. Evaluation on labeled test set

**Not possible.** `dataset/annotations.csv` lists **643** dentist-style rows:

| Split | Rows |
| --- | ---: |
| train | 440 |
| val | 110 |
| test | 93 |

| ICDAS | Rows in CSV |
| --- | ---: |
| 0 | 76 |
| 1 | **145** |
| 2 | **118** |
| 3 | **121** |
| 4 | 183 |

**Pixels on disk:** `dataset/{train,val,test}/0–4/` = **0 files**. `data_icdas/final/` = **0 labeled JPGs**. 16 files under `dataset/excluded/` are grades **5/6** (must not be used for 0–4).

| Metric | Value |
| --- | --- |
| accuracy | n/a |
| macro F1 | n/a |
| per-class P/R | n/a |
| confusion matrix | n/a |
| test samples/class | **0 on disk** |
| prediction distribution (labeled test) | n/a |

Classes **1, 2, 3 are present in the CSV**, absent as image files. The 5,676 auto-predictions were **not** used as ground truth.

## F. Action required

**MORE LABELED ICDAS DATA REQUIRED**

Do not continue Grad-CAM / report / UI on this classifier. Do not auto-label the 5,676 crops. Do not overwrite `deploy.keras`.

When 0–4 crop **pixels** matching `annotations.csv` (or a new dentist-labeled set with all five classes) are restored:

1. Train a **new** MobileNetV3+CBAM **5-class softmax** (`ordinal_regression: false`) into a **new** folder (e.g. `models/icdas_mobilenet_cbam_softmax_0_4/`).
2. Validate on a held-out labeled test split (confusion matrix must be able to show 1–3 if those classes exist).
3. Only then copy to `deploy.keras`.

Until then, `FIX MODEL/INFERENCE ARCHITECTURE` is a **requirement of the next train**, not a substitute for labels: keep production inference on **5-class softmax**; do not promote the current 4-output ordinal file.

## G. Files changed

- `docs/PROJECT_SCOPE.md` (new)
- `reports/ICDAS_CLASSIFIER_AUDIT.md` (this file)
- `reports/icdas_classifier_audit.json` (machine-readable audit)
- `tools/audit_icdas_classifier.py` (new)
- `README.md` (pipeline + FDI scope; no training)
- `docs/ARCHITECTURE.md` (high-level pipeline; no API wiring)

## H. Files not changed

- 420 originals: `fdi_detection_dataset/images/selected/`
- Batch 01 detection labels / YOLO `best.pt`
- `dataset/annotations.csv` and `dataset/` pixels (none to alter)
- `cropped_teeth/images/` (not overwritten)
- `models/deploy.keras`, `models/best.keras`
- FastAPI / Streamlit not wired for this audit
