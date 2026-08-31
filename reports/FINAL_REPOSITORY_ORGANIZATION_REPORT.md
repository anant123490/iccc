# Final repository organization report

**Project:** CCC AI Dentist Camera 2.0  
**Date:** 2026-08-27  
**Mode:** Reorganize in place. No deletions. No ICDAS training. No auto-labeling. No `deploy.keras` replacement.

## 1. Before structure (messy root)

Typical pre-cleanup layout (see `reports/REPOSITORY_ORGANIZATION_AUDIT.md`):

```text
backend/  fronted/  ml/  dataset/  data_icdas/  cropped_teeth/
fdi_detection_dataset/  models/{deploy.keras, tooth_detector_batch01, icdas_*}
annotation_batches/  annotation_project/  tools/  scripts/  reports/
FDI_*.md  STAGE3*.md  runs/  predictions/  crm_backend/
```

A first move pass (logged in `archive/review_required/phase4_move_log.txt`) already relocated apps, models, crops, ICDAS CSV, and FDI markdown.

## 2. After structure

See `docs/PROJECT_STRUCTURE.md`. Active code: `app/`, `ml/`, `tools/`. Data: `data/` plus canonical `fdi_detection_dataset/`. Models: `models/detection/` and `models/icdas/{current,historical}/`. History: `archive/`.

## 3. Files moved (this completion pass + prior pass)

Prior pass highlights:

- `backend/` → `app/backend/`
- `fronted/` → `app/frontend/`
- `cropped_teeth/` → `data/tooth_crops/generated/`
- `dataset/annotations.csv` → `data/icdas/annotations/annotations.csv`
- `models/tooth_detector_batch01/` → `models/detection/tooth_detector_batch01/`
- `models/deploy.keras` + `best.keras` → `models/icdas/historical/stale_ordinal_4output/` (files preserved, not replaced)
- FDI reports and `fdi_mapping/` → `archive/out_of_scope/fdi/`
- Ultralytics `runs/` and failed YOLO runs → `archive/experiments/`

This pass:

| From | To |
|------|-----|
| `scripts/*.py` | `tools/ingest/` |
| `tools/train_caries_detector.py` | `archive/experiments/caries_lesion/` |
| `tools/prepare_public_caries_yolo.py` | `archive/experiments/caries_lesion/` |

Empty future folders added under `data/detection/`, `ml/detection/`, `ml/icdas/`, `models/icdas/current/`, `tests/`.

## 4. Files intentionally left in place

| Path | Reason |
|------|--------|
| `fdi_detection_dataset/` | YOLO `path:`, cropper `DEFAULT_SOURCE`, annotation `relative_path` |
| `annotation_batches/`, `annotation_project/` | Tools and CVAT docs |
| `crm_backend/` | Separate app; not imported by camera API |
| `ml/.git/` | Nested repository |
| `assets/` | Junction |
| `data_external/` | Gitignored external dump |
| `app/backend/app/caries_detector.py`, `caries_pipeline.py` | Live FastAPI imports |
| `tools/crop_teeth.py` | Still used by toolkit / selftest |

## 5. Files archived

FDI search markdown/JSON, mapping stub, `.bak_pre_caries_pipeline` sources, old YOLO runs, TFJS/historical ICDAS experiments, lesion train scripts, empty leftover `dataset/` directory.

## 6. Dependencies updated

- Docker / compose / CI already targeted `app/backend` and historical keras paths
- `ml/train.py` default `overwrite_root_checkpoints` → **false**; outputs under `models/icdas/current/`
- `ml/configs/default.yaml` `dataset_root: data/icdas`
- `ml/scripts/sync_annotations.py` writes `data/icdas/annotations/annotations.csv` and will not wipe it when folders are empty
- `ml/src/tooth_cropping.py` already used `models/detection/...` and `data/tooth_crops/generated/`
- Ingest scripts `PROJECT_ROOT` after move to `tools/ingest/`
- Archived caries scripts `ROOT` → `parents[3]`
- `tools/train_tooth_detector_batch01.py` refuses overwrite unless `--force-retrain-batch01`
- New `tools/train_tooth_detector_new_batch.py` writes `tooth_detector_batchNN` only
- Docs/README/SETUP/DEPLOYMENT/DATASETS/TRAINING/ARCHITECTURE/tools README

## 7. Validation results

Checked on 2026-08-27:

| Check | Result |
|-------|--------|
| 420 originals | **420** JPGs in `fdi_detection_dataset/images/selected/` |
| Batch 01 images | 46 / 6 / 8 |
| Verified boxes | **767** YOLO lines |
| `best.pt` | exists (`models/detection/tooth_detector_batch01/weights/best.pt`) |
| Tooth crops | **5676** images in `data/tooth_crops/generated/images/` |
| ICDAS CSV | **643** rows at `data/icdas/annotations/annotations.csv` |
| ICDAS pixels invented | **No** |
| Auto ICDAS labels created | **No** |
| `deploy.keras` replaced | **No** (same stale file, new folder + `STALE.md`) |
| FastAPI | `app/backend/app/main.py` |
| Streamlit | `app/frontend/streamlit_app.py` |
| ML train | `ml/train.py` |
| FDI material deleted | **No** |
| Historical models deleted | **No** |

Syntax check of moved ingest / new train script: run after this report.

## 8. Remaining messy / high-risk items

- `fdi_detection_dataset/` name vs `data/detection/` (documented; do not rename until a dedicated path PR)
- `ml/.git` nested repo
- `assets/` junction
- `crm_backend/` unwired
- Live `caries_pipeline` vs whole-tooth product story
- ICDAS class folders still empty of pixels
- Ultralytics `args.yaml` may still store old absolute Windows paths (metadata only)

## 9. How to add new tooth-detection data

See `data/detection/README.md`: `raw_images/` + `annotations/` → new `batches/batch02/` → `python tools/train_tooth_detector_new_batch.py --batch 02`.

## 10. How to add new ICDAS data

See `data/icdas/README.md`: clinician-confirmed tooth image → `train|val|test/<0-4>/` → sync CSV → train softmax **later**.

## 11. What must NOT be changed

- Do not delete or flatten Batch 01
- Do not overwrite `tooth_detector_batch01/weights/best.pt` with Batch 02
- Do not replace stale `deploy.keras`
- Do not auto-label 5,676 crops
- Do not remap ICDAS 5/6 → 4
- Do not add FDI to the active pipeline
- Do not move `fdi_detection_dataset/` until every YAML/JSON/Python path is updated together

## 12. Current project status

**Detection:** usable Batch 01 model and dataset.  
**ICDAS:** code ready (5-class softmax); **data pixels missing**; production keras **stale**.  
**Apps:** FastAPI + Streamlit relocated under `app/`.  
**Camera loop:** conceptually defined; ICDAS grade not production-valid until a new softmax model is trained on real labels.
