# Project current state — CCC AI Dentist Camera 2.0

Date: 2026-08-26  
Audit only. No application behavior was changed in this document’s generation.

Approximate implementation: ICDAS classification + API + Streamlit + Groq exist; **caries localization detector was missing**.

---

## What already works

| Component | Location | Status |
| --- | --- | --- |
| ICDAS 0–4 classifier | `ml/src/model.py` (MobileNetV3Small + CBAM + 5-class softmax) | Working training/export path |
| Production checkpoint | `models/deploy.keras` (also `models/best.keras`) | Present on disk |
| Inference | `backend/app/inference.py` `InferenceEngine` | Upload → 224×224 [0,255] → softmax → confidence 0.55 |
| Grad-CAM | `InferenceEngine.generate_gradcam` / `explain`; also `ml/src/gradcam.py` | Working on classifier |
| FastAPI | `backend/app/main.py` | `/api/v1/predict`, health, history, stats, report |
| Groq | `backend/app/groq_service.py` | Structured text from ICDAS fields; local fallback if no key |
| Streamlit | `fronted/streamlit_app.py` (folder name is `fronted`) | Upload + `st.camera_input` → `/api/v1/predict` |
| ICDAS actions | `backend/app/icdas_actions.py` | Clinical copy for grades 0–4 |
| Zenodo caries dump | `data_external/detection/` | **6,265** RGB JPG already downloaded |
| 420 camera-domain copies | `fdi_detection_dataset/images/selected/` | Present; **no tooth boxes** |

---

## What partially works

- **ROI:** `use_roi` exists as a 5% crop, **off** in production (`config.py`). Not a learned detector.
- **Image quality:** Stage 3A quality heuristics exist in `fdi_detection_dataset/metadata/_build_stage3a.py`; **not** used at inference.
- **Live camera:** Snapshot via Streamlit camera widget, **not** video-frame sampling.
- **ICDAS `dataset/`:** `annotations.csv` historically listed crops; train folders were empty in Stage 2E. Do not treat as the 6k detection set.
- **Whole-tooth / FDI:** Stage 3A–3C infrastructure only; not implemented in the app.

---

## What was missing (before this increment)

- Learned **caries/decay-region** detector (`d`/`D`)
- Detector → crop → ICDAS per region
- Explicit no-detection vs whole-image fallback
- Isolated `models/caries_detector/`

---

## What must not be changed

- `fdi_detection_dataset/images/selected/` (420 originals)
- `dataset/` ICDAS crops/labels
- Existing ICDAS `.keras` files (no overwrite)
- Do not map `d`/`D` → ICDAS
- No CVAT, no FDI fabrication, no SegmentAnyTooth / PhysioNet

---

## Public dataset (do not re-download)

Already on disk: Zenodo `10.5281/zenodo.14827784` under `data_external/detection/`.  
See `STAGE2C_ZENODO_DETECTION_REPORT.md`. Classes: **`D`** (YOLO 0, permanent decay), **`d`** (YOLO 1, primary decay).
