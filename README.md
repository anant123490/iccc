# CCC AI Dentist Camera 2.0

Research prototype: **ICDAS 0–4** caries severity from RGB intraoral photographs.

This is **not** a replacement for professional dental diagnosis.

**FDI tooth numbering is out of scope.** Do not add FDI labels, models, or pipeline steps.

## What is completed

- Patient-facing loop (concept): registration / visit → RGB photo → quality check → whole-tooth detection → crops → ICDAS → Grad-CAM → report
- **Tooth detector (Batch 01):** YOLO11n, 46/6/8 images, **767** verified tooth boxes, weights at `models/detection/tooth_detector_batch01/weights/best.pt`
- **420** original RGB intraoral photographs
- Previous 5,676 detector-generated crops and their 420 overlays were cleared; the 420 source photographs remain
- FastAPI (`app/backend/`) and Streamlit (`app/frontend/`)
- MobileNetV3 + CBAM training code for **5-class softmax**

Approximate Batch 01 held-out detection: Precision 0.700, Recall 0.726, F1 0.712, mAP50 0.718, mAP50-95 0.282 (tiny test set).

## What is currently blocked

- The previous ICDAS annotations and classifier artifacts were cleared; new clinician-confirmed 0–4 images and labels are needed
- **Do not train** a new ICDAS model yet
- Detector crops are not ICDAS ground truth; require clinician labels before training
- No ICDAS checkpoint is present; the stale 4-output ordinal weights were removed
- No valid production ICDAS classifier is currently available

## Where things live

| Question | Answer |
|----------|--------|
| Detection dataset (Batch 01) | `fdi_detection_dataset/` — **left in place** (code still uses this path). Historical name; **not** FDI labels |
| New detection photos | `data/detection/raw_images/` |
| New tooth boxes | `data/detection/annotations/` |
| Generated tooth crops | `data/tooth_crops/generated/` |
| New ICDAS tooth images | `data/icdas/raw/` then `data/icdas/train\|val\|test/0–4/` |
| Confirmed Admin training exports | `data/icdas/datasets/dataset_v001/` — patient-wise database assignments with tooth-crop `labels.csv` files; versions are immutable |
| ICDAS labels | Rebuild `data/icdas/annotations/annotations.csv` from new clinician-confirmed data |
| Valid detection model | `models/detection/tooth_detector_batch01/weights/best.pt` |
| ICDAS models | No ICDAS weights remain; `models/icdas/current/` is reserved for a future evaluated model |
| Out of scope | FDI (`archive/out_of_scope/fdi/`), ICDAS 5–6 (`data/icdas/excluded/`) |
| How to add Batch 02 | See `data/detection/README.md` then `python tools/train_tooth_detector_new_batch.py --batch 02` |
| How to add ICDAS data | See `data/icdas/README.md` — clinician-confirmed 0–4 only |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy app\backend\.env.example app\backend\.env
```

```bash
cd app/backend
uvicorn app.main:app --reload --port 8000
```

```bash
streamlit run app/frontend/streamlit_app.py
```

ICDAS training (only after labeled pixels exist):

```bash
python ml/train.py --config ml/configs/default.yaml
```

## Tests

```bash
cd ml && pytest tests/ -v
cd app/backend && pytest tests/ -v
```

## Docs

- `docs/PROJECT_STRUCTURE.md` — full tree
- `docs/DATASET_WORKFLOW.md` — detection vs ICDAS
- `docs/PROJECT_SCOPE.md` — what is in / out of scope
- `reports/FINAL_REPOSITORY_ORGANIZATION_REPORT.md` — this reorganization
