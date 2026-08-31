# Windows setup — one ML environment

CCC AI Dentist Camera 2.0 uses **one Python virtual environment** for FastAPI, YOLO, TensorFlow, and Streamlit.

Do **not** create three venvs and install TensorFlow/YOLO three times.

```
.venv  (single env)
 ├── backend   uvicorn FastAPI  :8000   ← YOLO + TF + DB live here
 ├── patient   Streamlit        :8502   ← HTTP client only
 └── admin     Streamlit        :8503   ← HTTP client only
```

## 1. Create the backend environment (once)

In PowerShell from the repo root:

```powershell
cd "C:\Users\anant\OneDrive\Desktop\icdas project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy app\backend\.env.example app\backend\.env
```

Edit `app\backend\.env`:

- `GROQ_API_KEY` (optional, reports)
- `ICDAS_ADMIN_PASSWORD` (admin login; default `changeme`)
- `ICDAS_ALLOW_TRAIN=0` until you are ready to train ICDAS
- Leave `ICDAS_MODEL_PATH` pointing at `models/icdas/current/deploy.keras` (file appears only after SET ACTIVE)

Tooth Detector V2 weights stay at:

`models/detection/tooth_detector_v2/weights/best.pt`

Do not retrain them.

## 2. Start the backend

```powershell
.\scripts\start_backend.ps1
```

or `scripts\start_backend.bat`

API: http://127.0.0.1:8000  
Health: http://127.0.0.1:8000/api/v1/portal/health

## 3. Start the patient frontend

New terminal, same `.venv`:

```powershell
.\scripts\start_patient.ps1
```

http://127.0.0.1:8502

Talks only to `/api/v1/...`. No local model loading.

## 4. Start the admin frontend

New terminal, same `.venv`:

```powershell
.\scripts\start_admin.ps1
```

http://127.0.0.1:8503

Login uses `ICDAS_ADMIN_PASSWORD`. Upload / box review / ICDAS 0–4 labeling / BUILD DATASET all call the backend.

## 5. ICDAS training (later, not now)

Training is **off** until `ICDAS_ALLOW_TRAIN=1` is set on the **backend** process and the API is restarted.

When enabled, Admin → Training starts `ml/train.py` as a backend subprocess (MobileNetV3 + CBAM). It does not freeze Streamlit and does not retrain YOLO.

Models are written to `models/icdas/v1`, `v2`, … SET ACTIVE copies a keras file into `models/icdas/current/deploy.keras` for local inference only (not cloud deploy).
