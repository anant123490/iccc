# Setup Guide

## 1. Clone and environment

```bash
cd "icdas project"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy app\backend\.env.example app\backend\.env
```

Put `GROQ_API_KEY` in `app/backend/.env`. Never commit `.env`.

## 2. ICDAS images (when you have clinician labels)

```bash
python ml/scripts/setup_dataset.py
# Copy labeled tooth images into data/icdas/train|val|test/0..4/
# Do not copy ICDAS 5 or 6 into class 4 (use data/icdas/excluded/).
python ml/scripts/sync_annotations.py
python ml/scripts/validate_dataset.py --allow-empty
```

Optional ingest:

```bash
python tools/ingest/download_datasets.py --dataset dental_caries
python tools/ingest/preprocess_dataset.py
```

Training is **blocked** until real pixels exist. See `data/icdas/README.md`.

## 3. Train ICDAS (later)

```bash
python ml/train.py --config ml/configs/default.yaml
```

Writes under `models/icdas/current/<experiment>/`. Does not overwrite historical keras unless you set `overwrite_root_checkpoints: true`.

## 4. Run apps (one venv)

See `docs/WINDOWS_SETUP.md`. From repo root after `.\.venv\Scripts\Activate.ps1`:

```powershell
.\scripts\start_backend.ps1
.\scripts\start_patient.ps1
.\scripts\start_admin.ps1
```

- API: http://127.0.0.1:8000
- Patient: http://127.0.0.1:8502
- Admin: http://127.0.0.1:8503

## 5. Tooth detector

Batch 01 weights: `models/detection/tooth_detector_batch01/weights/best.pt`

New data: `data/detection/README.md`.
