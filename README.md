# Offline AI-Based Dental Caries Detection (ICDAS Classification)

Production-grade Progressive Web App for **offline** dental caries severity detection from intraoral smartphone photos using Edge AI and ICDAS grading.

> **Disclaimer:** This tool is for clinical decision support and is not a substitute for professional diagnosis.

## Features

| Feature | Description |
|---------|-------------|
| **Offline PWA** | Full inference after install — no internet required |
| **ICDAS 0–6** | Ordinal regression with confidence scores |
| **Explainability** | Grad-CAM heatmaps + lesion contour overlays |
| **Edge AI** | MobileNetV3-Small + CBAM, TFLite/TF.js export (<20MB target) |
| **Privacy** | Local-only storage, AES encryption, consent screen |
| **History** | IndexedDB patient scan history & progression tracking |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  React PWA      │────▶│  TensorFlow.js   │     │  IndexedDB      │
│  (Camera/UI)    │     │  (Edge Inference)│     │  (History)      │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │ optional online
         ▼
┌─────────────────┐     ┌──────────────────┐
│  FastAPI        │────▶│  Keras / TFLite  │
│  Backend        │     │  + Grad-CAM      │
└─────────────────┘     └──────────────────┘
         ▲
         │ train / export
┌─────────────────┐
│  ML Pipeline    │
│  (MobileNetV3)  │
└─────────────────┘
```

## Project Structure

```
icdas-project/
├── frontend/          # React + TypeScript PWA
├── backend/           # FastAPI inference API
├── ml/                # Training, preprocessing, export
├── dataset/           # train/val/test + annotations.csv
├── models/            # Exported weights (gitignored)
├── docker/            # Dockerfiles
├── docs/              # Guides, paper draft, slides
├── scripts/           # Dataset download utilities
└── .github/workflows/ # CI/CD
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) Docker

### 1. Dataset Setup

**15 WhatsApp clinical images are already imported** into `dataset/` (train/val/test).

```bash
# Re-import or add more images from assets/
python scripts/import_whatsapp_images.py

# Create empty folder structure for additional data
python ml/scripts/setup_dataset.py

# Download public datasets (see docs/DATASETS.md)
python scripts/download_datasets.py --dataset dental_caries
```

See `dataset/README.md` for label details and split info.

### 2. Train Model

```bash
cd ml
pip install -r requirements.txt
python train.py --config configs/default.yaml
python export.py --checkpoint ../models/best.keras --quantize
```

### 3. Copy Model to PWA

```bash
cp models/tfjs_model/* frontend/public/models/
```

### 4. Run Frontend (Offline PWA)

```bash
cd frontend
npm install
npm run dev        # Development
npm run build      # Production PWA
```

### 5. Run Backend (Optional)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 6. Docker

```bash
docker compose up --build
```

## ICDAS Classes

| Grade | Description | Suggested Action |
|-------|-------------|------------------|
| 0 | Sound tooth | Monitor |
| 1 | Initial lesion | Monitor + fluoride |
| 2 | Distinct visual change | Fluoride treatment |
| 3 | Localized enamel breakdown | Restoration needed |
| 4 | Underlying dentin | Restoration needed |
| 5 | Distinct cavity with dentin | Restoration needed |
| 6 | Extensive distinct cavity | Restoration needed |

## Configuration

Edit `ml/configs/default.yaml`:

```yaml
num_classes: 7          # ICDAS 0-6 (set to 5 for 0-4)
use_attention: cbam     # cbam | se | none
ordinal_regression: true
image_size: 224
```

## API Documentation

With backend running: http://localhost:8000/docs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Image → ICDAS + heatmap |
| `/api/v1/health` | GET | Health check |
| `/api/v1/model/info` | GET | Model metadata |

## Testing

```bash
# ML unit tests
cd ml && pytest tests/ -v

# Backend tests
cd backend && pytest tests/ -v

# Frontend tests
cd frontend && npm test
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Inference latency | <1s (mobile) |
| Model size | <20MB |
| Offline | 100% after PWA install |

## License

MIT — See LICENSE. Dataset licenses vary; see `docs/DATASETS.md`.

## Citation

If you use this project in research, see `docs/RESEARCH_PAPER_DRAFT.md`.
