# AI-Based Dental Caries Detection and ICDAS Severity Classification

Research prototype for **ICDAS 0–4** caries severity classification from intraoral photographs.

> **Disclaimer:** This is an AI decision-support / research prototype. It is **not** a replacement for professional dental diagnosis.

## Supported classes

```text
ICDAS 0–4
```

The previous 7-class (ICDAS 0–6) output head is incompatible and is not used.
ICDAS 5 and 6 images are **not** remapped to ICDAS 4.

## ML architecture

```text
MobileNetV3-Small
+
CBAM
+
Ordinal Regression
+
Grad-CAM
```

Five classes use **four** ordinal thresholds. The CNN assigns the ICDAS grade; Groq only writes an explanation.

## Backend

```text
FastAPI
PostgreSQL (or SQLite locally)
SQLAlchemy
```

JWT authentication is not enabled in this inference API. Secrets stay in `.env`.

## AI

```text
Groq API
```

Groq never overrides the model’s ICDAS grade.

## Frontend

```text
Streamlit
```

## Pipeline

```text
Image
 ↓
Preprocessing
 ↓
MobileNetV3-Small
 ↓
CBAM
 ↓
Ordinal Regression
 ↓
ICDAS 0–4
 ↓
Grad-CAM
 ↓
Groq Report
 ↓
Frontend
```

## Project structure

```text
iccc/
├── backend/          FastAPI inference, history, Groq reports
├── ml/               Training, model, dataset, Grad-CAM
├── dataset/          train/val/test folders for classes 0–4
├── models/           best.keras, deploy.keras (5-class)
├── fronted/          Streamlit app
└── docs/
```

## Quick start

### 1. Environment

```bash
cd iccc
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy backend\.env.example backend\.env
```

Put `GROQ_API_KEY` in `backend/.env`. Never commit `.env`.

Optional PostgreSQL:

```env
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/icdas
```

If `DATABASE_URL` is omitted, SQLite `icdas_predictions.db` is used.

### 2. Dataset (ICDAS 0–4)

```bash
python ml/scripts/setup_dataset.py
python ml/scripts/split_dataset.py   # 70/15/15, excludes 5/6 without remapping
python ml/scripts/sync_annotations.py
python ml/scripts/validate_dataset.py
```

Expected layout:

```text
dataset/train/{0,1,2,3,4}
dataset/val/{0,1,2,3,4}
dataset/test/{0,1,2,3,4}
```

Augmentation is applied **only** during training.

### 3. Train a fresh 5-class model

```bash
python ml/train.py --config ml/configs/default.yaml
```

Writes `models/best.keras` and `models/deploy.keras`. Do not reuse a 7-class checkpoint.

### 4. Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 5. Frontend

```bash
streamlit run fronted/streamlit_app.py
```

## Configuration

`ml/configs/default.yaml`:

```yaml
num_classes: 5
ordinal_regression: true
image_size: 224
backbone: mobilenet_v3_small
use_attention: cbam
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` or `/predict` | POST | Image → ICDAS 0–4 + Grad-CAM + report |
| `/api/v1/report` | POST | Groq explanation for a **model** grade |
| `/api/v1/history` | GET | Stored predictions |
| `/api/v1/stats` | GET | Dashboard analytics |
| `/api/v1/health` | GET | API / model / database / Groq status |
| `/api/v1/model/info` | GET | Model metadata |

## Testing

```bash
cd ml && pytest tests/ -v
cd backend && pytest tests/ -v
python -c "from ml.src.model import build_model; m=build_model(num_classes=5, image_size=224, attention_type='cbam', ordinal_regression=True); print(m.output_shape)"
python ml/scripts/validate_dataset.py
```

## License

MIT — See LICENSE. Dataset licenses vary; see `docs/DATASETS.md`.
