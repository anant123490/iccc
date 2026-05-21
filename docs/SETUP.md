# Setup Guide

## 1. Clone & Environment

```bash
cd "icdas project"

# Python virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r ml/requirements.txt
pip install -r backend/requirements.txt

# Node.js frontend
cd frontend && npm install
```

## 2. Dataset

```bash
python ml/scripts/setup_dataset.py
# Copy labeled images into dataset/train|val|test/0..6/
python ml/scripts/sync_annotations.py   # required so annotations.csv matches folders
python scripts/download_datasets.py --dataset dental_caries  # optional
python scripts/preprocess_dataset.py --input dataset/raw --output dataset
```

For ~2000 ICDAS-labeled images: place files under `train/`, `val/`, and `test/` class folders, then always run `sync_annotations.py` before training.

## 3. Train

```bash
cd ml
python train.py --config configs/default.yaml
python export.py --checkpoint ../models/best.keras --quantize
# If TF.js export succeeds:
Copy-Item -Recurse ..\models\tfjs_model\* ..\frontend\public\models\
# If TF.js export fails on Windows, use the backend API for real inference (see section 4).
```

## 4. Run

```bash
# PWA (offline)
cd frontend && npm run dev

# API (optional)
cd backend && uvicorn app.main:app --reload
```

## 5. Install PWA on Mobile

1. Open `http://<your-ip>:5173` on Android/iPhone
2. Chrome: Menu → "Install app" / Safari: Share → "Add to Home Screen"
3. App works offline after first load
