# Deployment Guide

## PWA Production Build

```bash
cd frontend
npm run build
# Serve dist/ with any static host or Docker
```

## Docker Full Stack

```bash
docker compose up --build
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

## Edge Optimization Checklist

- [ ] Run `python ml/export.py --quantize`
- [ ] Verify `export_report.json`: p95 < 1000ms, size < 20MB
- [ ] Copy TF.js model to `frontend/public/models/`
- [ ] Test on low-end Android (Chrome DevTools throttling)

## Mobile TFLite (Native Wrapper)

Use `models/model.tflite` with:
- Android: TensorFlow Lite Interpreter
- iOS: TensorFlowLiteSwift

## Security

- All inference local by default
- AES-GCM encryption for IndexedDB (Settings → Encrypt)
- Consent screen on first launch
- No telemetry or cloud upload

## CI/CD

GitHub Actions runs on push to `main`:
- ML unit tests
- Backend API tests
- Frontend build + lint
- Docker image build
