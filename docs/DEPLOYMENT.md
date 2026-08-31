# Deployment Guide

## Streamlit + FastAPI

```bash
cd app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

streamlit run app/frontend/streamlit_app.py --server.port 8501
```

## Docker

```bash
docker compose up --build
```

- UI: http://localhost:8501
- Backend: http://localhost:8000

Compose mounts `models/` read-only. ICDAS production weights are `models/icdas/current/deploy.keras` (5-class softmax). If that file is missing, the API reports ICDAS as not trained/deployed. Historical stale ordinal checkpoints are not used.

## Optional export

```bash
python ml/export.py --checkpoint ../models/icdas/current/<experiment>/best.keras --quantize
```

## Security

- Groq only explains a grade already produced by the CNN.
- Secrets stay in `.env`.
- This API does not currently require JWT.

## CI

GitHub Actions: `ml/tests`, `app/backend/tests`, Docker build of the backend image.
