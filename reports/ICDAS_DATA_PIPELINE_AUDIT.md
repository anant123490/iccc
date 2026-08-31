# ICDAS data pipeline audit

Date: 2026-08-26

This pass inspected the live repo and added an ICDAS **v2 labeling** tree. It did **not** rebuild the product, did **not** train, and did **not** overwrite existing ICDAS assets.

---

## Existing ICDAS pipeline (reused)

| Piece | Location | Notes |
| --- | --- | --- |
| MobileNetV3-Small + CBAM | `ml/src/model.py`, `ml/src/attention.py` | 5-class softmax; ordinal optional |
| ICDAS 0–4 constants | `ml/src/icdas.py` | 5 and 6 out of scope |
| Grad-CAM | `ml/src/gradcam.py` | Reused by backend inference |
| Training | `ml/train.py` + `ml/configs/default.yaml` | Class weights, focal/ordinal options |
| Dataset loader | `ml/src/dataset.py` | Reads `dataset/train\|val\|test/0-4` |
| Existing labels CSV | `dataset/annotations.csv` | **Untouched** |
| Checkpoints | `models/icdas_mobilenet_cbam*` | **Not overwritten** |
| FastAPI | `backend/app/main.py`, `inference.py` | `/api/v1/predict` |
| Groq | `backend/app/groq_service.py` | Env key; does not change grades |
| Image quality | `backend/app/image_quality.py` | Existing |
| History DB | `backend/app/db_models.py` | `PredictionRecord` (not Patient/Visit tables) |
| Streamlit product UI | `fronted/streamlit_app.py` | **Untouched**; includes Detection/Analysis/History |
| Old labeling UI | `tools/label_icdas.py` | Crops from `cropped_teeth/`; still valid |
| Cropper | `tools/crop_teeth.py` | Region crops from **existing boxes** — must **not** be pointed at d/D for ICDAS |
| Dataset builder (legacy) | `tools/build_dataset.py` | Writes **`dataset/`** — do not use for v2 |
| Lesion detector | `backend/app/caries_detector.py` | YOLO **D/d only** — not ICDAS, not whole-tooth |
| Public RGB + lesion boxes | `data_external/detection/` | ~6265 JPG; 2227 with d/D; 4038 unannotated — **not ICDAS** |
| Personal 420 | `fdi_detection_dataset/images/selected/` | Source images; **not modified** |

---

## Files that will NOT be modified (this design)

- `dataset/` (including `annotations.csv`)
- existing Keras trees under `models/icdas_mobilenet_cbam*`
- `ml/src/*` architecture / Grad-CAM
- `backend/` (except unused)
- `fronted/streamlit_app.py`
- `fdi_detection_dataset/images/selected/`
- `data_external/detection/` images and lesion XML

`ml/train.py` was given two **additive** config keys only: `dataset_root` (default still `dataset/`) and `overwrite_root_checkpoints` (default still true so old `default.yaml` behavior is unchanged). `ml/configs/icdas_v2.yaml` sets `overwrite_root_checkpoints: false`.

---

## Created (v2)

- `data_icdas/` source manifest, crop pool, labels, `final/`
- `tools/icdas_v2_lib.py`
- `tools/inspect_icdas_data.py`
- `tools/ingest_icdas_public.py`
- `tools/import_icdas_images.py`
- `tools/register_icdas_crops.py`
- `tools/icdas_labeling_app.py` (ICDAS Labeling Studio)
- `tools/build_icdas_dataset.py` → `data_icdas/final` only
- `tools/icdas_label_qc.py`
- `tools/evaluate_icdas.py`
- `tools/build_active_learning_queue.py`
- `tools/run_icdas_inference.py`
- `ml/configs/icdas_v2.yaml`
- `tools/test_icdas_v2_pipeline.py`

---

## Tooth crops

**Unavailable automatically.** There is no verified whole-tooth detector in-repo. The caries YOLO must not generate ICDAS crops. Labeling Studio waits for files in `data_icdas/crops/` + `register_icdas_crops.py`.

---

## Training / metrics this pass

**Not run.** Zero dentist ICDAS v2 labels. Exact class counts in `data_icdas/final` are all **0**. Exact v2 metrics: **n/a**.

Existing product ICDAS model remains whatever is already under `models/icdas_mobilenet_cbam*`.
