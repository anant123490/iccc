# Repository organization audit — CCC AI Dentist Camera 2.0

**Mode:** AUDIT ONLY (2026-08-27). No files were moved, deleted, renamed, or modified except this file and `reports/repository_organization_audit.json`.

**Active pipeline:** RGB intraoral photo → image quality → whole-tooth detection → tooth crops → ICDAS 0–4 → Grad-CAM → clinical report.

**FDI:** Completely out of scope. Do not reintroduce. Historical FDI material should be isolated later, not deleted.

Machine-readable companion: `reports/repository_organization_audit.json`.

Categories: **A** ACTIVE · **B** ACTIVE DATA · **C** ACTIVE MODEL · **D** ACTIVE CODE · **E** DOCUMENTATION · **F** EXPERIMENTAL · **G** HISTORICAL · **H** OUT_OF_SCOPE · **I** DUPLICATE · **J** GENERATED · **K** TEMPORARY · **L** UNKNOWN

---

## 1. Current repository tree summary

```text
icdas project/
  README.md, LICENSE, requirements.txt, docker-compose.yml, .gitignore
  yolo11n.pt                          # COCO init for YOLO train
  STAGE3*.md, TOOTH_ANNOTATION_*.md, DATASET_REPORT.md, FDI_*.md, …
  .github/workflows/ci.yml
  docs/                               # scope, architecture, setup, training
  ml/                                 # ICDAS train + crop/predictor  (+ nested .git)
  backend/                            # FastAPI ICDAS + quality + Groq (+ caries experiment)
  fronted/                            # Streamlit (typo folder name)
  crm_backend/                        # separate CRM; not imported by ICDAS app
  tools/                              # YOLO train/crop, ICDAS label/audit CLIs
  scripts/                            # legacy ingest
  docker/
  models/                             # YOLO best.pt; stale deploy.keras; old folds
  fdi_detection_dataset/              # 420 RGB + Batch01 YOLO dataset (name is historical)
  annotation_batches/  annotation_project/
  cropped_teeth/                      # 5676 crops — NOT ICDAS GT
  predictions/                        # YOLO candidates + invalid ICDAS CSV
  dataset/                            # annotations.csv 643; pixels MISSING
  data_icdas/                         # empty crop/final scaffolding
  data_external/                      # Zenodo lesion RGB (gitignored)
  labels/  labels.csv/                # empty / suspicious empty dir
  reports/  runs/  assets/ (junction)
  .venv/ .venv-streamlit/ .venv311/ .pytest_cache/
```

No Jupyter notebooks found.

---

## Significant items (path · type · category · purpose · status · code refs · keep? · destination · risk)

| Path | Type | Cat | Purpose | Status | Referenced | Keep active | Recommended destination | Move risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `README.md` | file | E | Overview + pipeline | Current | No | Yes | root | low |
| `docs/PROJECT_SCOPE.md` | file | E | No FDI; camera pipeline | Current | No | Yes | `docs/` | low |
| `docs/ARCHITECTURE.md` | file | E | Softmax 5 intended | Current | No | Yes | `docs/` | low |
| `docs/TRAINING.md` `SETUP.md` `DEPLOYMENT.md` | file | E | Ops docs | Current | No | Yes | `docs/` | low |
| `requirements.txt` | file | A | Deps; CI | Current | Yes | Yes | root | **high** |
| `docker-compose.yml` `docker/` | file/dir | A | 5-class env; images | Current | Yes | Yes | root / `docker/` | **high** |
| `.github/workflows/ci.yml` | file | A | pytest ml+backend | Current | Yes | Yes | `.github/` | **high** |
| `ml/` | dir | D | Train/preprocess/CBAM/crop/predictor | Current; nested git | Yes | Yes | `ml/` | **high** |
| `ml/src/model.py` | file | D | Softmax 5 default; optional ordinal | Current | Yes | Yes | `ml/src/` | **high** |
| `ml/src/icdas.py` | file | D | Grades 0–4 only | Current | Yes | Yes | `ml/src/` | **high** |
| `ml/configs/default.yaml` | file | A | `ordinal_regression: false` | Current | Yes | Yes | `ml/configs/` | **high** |
| `ml/configs/icdas_v2.yaml` | file | F | v2 train; no overwrite root keras | Empty `data_icdas/final` | Yes | Yes | `ml/configs/` | medium |
| `ml/src/tooth_cropping.py` | file | D | YOLO crops | Current | Yes | Yes | `ml/src/` | **high** |
| `ml/src/icdas_predictor.py` | file | D | Loads `deploy.keras` | Model not prod-ready | Yes | Yes | `ml/src/` | **high** |
| `ml/src/gradcam.py` | file | D | Grad-CAM | Next stage | Yes | Yes | `ml/src/` | medium |
| `ml/.git/` | dir | L | Nested repo | Unexpected | No | Review | Do not move yet | **high** |
| `backend/app/inference.py` | file | D | Rejects 4-output ordinal | Mismatch vs file on disk | Yes | Yes | `backend/app/` | **high** |
| `backend/app/image_quality.py` | file | D | Blur/brightness | Current | Yes | Yes | `backend/app/` | **high** |
| `backend/app/groq_service.py` | file | D | Report text | Current | Yes | Yes | `backend/app/` | **high** |
| `backend/app/caries_detector.py` | file | F | Lesion `d`/`D` YOLO | Not whole-tooth ICDAS | Yes | Yes for now | later `archive/experimental/caries_lesion/` | medium |
| `backend/app/*.bak_pre_caries_pipeline` | file | G | Backups | Obsolete | No | No | `archive/backend_bak/` | low |
| `fronted/streamlit_app.py` | file | D | UI | Current | Yes | Yes | keep path (rename later) | **high** |
| `fronted/*.bak_*` | file | G | UI backup | Obsolete | No | No | `archive/frontend_bak/` | low |
| `crm_backend/` | dir | L | Separate CRM SaaS | Not imported by ICDAS | No | Yes until decided | later `apps/crm/` | medium |
| `tools/train_tooth_detector_batch01.py` | file | D | Train YOLO | Current | CLI | Yes | `tools/` | medium |
| `tools/run_tooth_cropping.py` | file | D | Crop 420 | Current | CLI | Yes | `tools/` | medium |
| `tools/label_icdas.py` `icdas_labeling_app.py` | file | D | Human ICDAS labels | Needed when pixels exist | CLI | Yes | `tools/` | medium |
| `tools/crop_teeth.py` | file | G | Old box cropper | Superseded by `tooth_cropping.py` | Weak | Keep until proven unused | `archive/tools_legacy/` | medium |
| `tools/train_caries_detector.py` | file | F | Lesion YOLO train | Not ICDAS | CLI | Isolate later | `archive/experimental/caries_lesion/` | low |
| `scripts/` | dir | F | WhatsApp/download ingest | Legacy | Weak | Yes | `archive/scripts/` | low |
| `models/tooth_detector_batch01/weights/best.pt` | file | C | **Valid** YOLO tooth detector | Current | Yes | Yes | keep | **high** |
| `yolo11n.pt` | file | C | Train init | Current | Yes | Yes | later `models/pretrained/` | medium |
| `models/deploy.keras` `best.keras` | file | G | Intended prod ICDAS; **(None,4) ordinal** | **Not prod-ready** | Yes | Yes until replaced | then `archive/models/icdas_ordinal_stale/` | **high** |
| `models/icdas_mobilenet_cbam/` | dir | G | 7-class ordinal folds | Obsolete | No | No | `archive/models/icdas_7class_ordinal/` | medium |
| `models/icdas_mobilenet_cbam_ordinal/` | dir | G | Ordinal keras | Obsolete | No | No | `archive/models/icdas_ordinal/` | medium |
| `models/icdas_mobilenet_cbam_5class_v2|v3|weighted/` | dir | G | Metrics-only experiments | No keras listed | No | No | `archive/models/icdas_5class_experiments/` | low |
| `models/archives/` `tooth_detector_batch01_run1_*` | dir | G | Old YOLO runs | Obsolete | No | No | `archive/models/yolo_runs/` | low |
| `models/tfjs_model/` | dir | F | TFJS shards | Export | No | No | `archive/models/tfjs/` | low |
| `fdi_detection_dataset/` | dir | B | 420 RGB + Batch01 YOLO | **Not FDI labels**; name historical | Yes | Yes | later `data/intraoral_rgb/` | **high** |
| `fdi_detection_dataset/images/selected/` | dir | B | 420 originals | Do not modify | Yes | Yes | keep | **high** |
| `fdi_detection_dataset/tooth_detector_batch01/` | dir | B | 46/6/8, 767 boxes | Verified | Yes | Yes | keep | **high** |
| `fdi_detection_dataset/annotations/fdi_mapping/` | dir | H | FDI mapping stub | Leftover | No | No | `archive/fdi_historical/` | low |
| `annotation_batches/` `annotation_project/` | dir | B | Human tooth-box QC | Batch01 done | Yes | Yes | keep | medium |
| `cropped_teeth/` | dir | J | 5676 crops | **Not ICDAS GT** | Yes | Yes | later `data/generated/cropped_teeth/` | medium |
| `predictions/labels/` `visualizations/` | dir | J | 360 YOLO candidates | Not GT | Weak | Yes | `data/generated/yolo_candidates/` | low |
| `predictions/icdas_predictions/` | dir | J | 0/4-only auto ICDAS | **Invalid clinical** | No | Archive later | `archive/generated/icdas_batch_invalid/` | low |
| `dataset/annotations.csv` | file | B | 643 ICDAS rows (1=145,2=118,3=121) | **Pixels missing** | Yes | Yes | restore files under `dataset/{split}/{class}/` | **high** |
| `dataset/train|val|test/` | dir | B | Intended ICDAS pixels | **0 files** | Yes | Yes | keep empty folders | **high** |
| `dataset/excluded/` | dir | H | 16× ICDAS 5/6 | Correctly out of 0–4 | No | Yes | stay excluded | medium |
| `data_icdas/` | dir | B | v2 pool/final | Empty crops | Yes | Yes | keep | medium |
| `data_external/detection/` | dir | F | Zenodo RGB + `d`/`D` | Not ICDAS/teeth GT | Yes | Isolate later | `archive/external/zenodo_lesion/` | medium |
| `labels/` | dir | B | Empty `.gitkeep` | Empty | Yes | Yes | merge later | low |
| `labels.csv/` | dir | L | Empty dir named like a CSV | Suspicious | No | Review | manual | low |
| `assets/` | junction | L | gitignored Cursor junction | External | No | No | do not touch | **high** |
| `runs/detect/` | dir | J | Ultralytics val dumps | Temp | No | No | archive/delete later | low |
| `reports/` | dir | E/J | Audits + stage reports | Mixed | No | Yes | split archive FDI reports | low |
| Root `FDI_*.md` `FDTooth_*.md` `RGB_FDI_*.md` `STAGE2E_RGB_FDI_*` | file | H | FDI searches | Historical | No | No | `archive/fdi_historical/` | low |
| `STAGE3C_SEED_TRAINING_REPORT.md` | file | I/G | Superseded by Batch01 train report | Old | No | No | archive | low |
| `.venv*` `.pytest_cache/` | dir | K | Local envs/cache | Local | No | No | gitignored | low |

---

## 2. Active project components

- **Scope/docs:** `docs/PROJECT_SCOPE.md`, `README.md`, `docs/ARCHITECTURE.md`, `TOOTH_ANNOTATION_GUIDELINES.md`, `STAGE3C_MANUAL_ANNOTATION.md`.
- **Quality:** `backend/app/image_quality.py`.
- **Detection:** `models/tooth_detector_batch01/weights/best.pt`, `fdi_detection_dataset/tooth_detector_batch01/`, `tools/train_tooth_detector_batch01.py`, `ml/src/tooth_cropping.py`.
- **Crops (generated, not labels):** `cropped_teeth/`.
- **ICDAS code (not a valid checkpoint):** `ml/train.py`, `ml/src/{model,icdas,preprocessing,dataset,gradcam,icdas_predictor}.py`, `ml/configs/default.yaml`, `backend/app/inference.py`.
- **ICDAS labels (CSV only):** `dataset/annotations.csv`.
- **UI/API (not rewired):** `backend/`, `fronted/streamlit_app.py`.
- **CI/deploy:** `.github/workflows/ci.yml`, `docker/`, `docker-compose.yml`.

## 3. Historical components

- 7-class / ordinal ICDAS: `models/icdas_mobilenet_cbam/` (`config.json` `num_classes: 7`), `models/icdas_mobilenet_cbam_ordinal/`, on-disk `deploy.keras`/`best.keras` (4-output `ordinal` layer).
- Failed/old YOLO runs: `models/archives/`, `models/tooth_detector_batch01_run1_adamw_lr0.01_collapsed/`.
- 5-class experiment folders with metrics only: `models/icdas_mobilenet_cbam_5class_v2|v3|weighted/`.
- `tools/crop_teeth.py` vs current `ml/src/tooth_cropping.py`.
- `*.bak_pre_caries_pipeline` in backend/fronted.
- Stage 2–3C seed reports superseded by later Batch 01 training report.

## 4. FDI / out-of-scope components

**Do not delete.** Isolate later:

- Root: `FDI_RGB_DATASET_SEARCH.md`, `FDI_RGB_FINAL_VERIFICATION.md`, `FDTooth_ACQUISITION_REPORT.md`, `RGB_FDI_DATASET_SEARCH_STAGE2D6.md`, `STAGE2E_RGB_FDI_FEASIBILITY_REPORT.md`.
- `reports/RGB_TOOTH_FDI_PUBLIC_DATASET_RANKING.md`, `reports/stage2d*_fdi_*`.
- `fdi_detection_dataset/annotations/fdi_mapping/` (README only).
- Folder **name** `fdi_detection_dataset/` is historical; contents are RGB + **tooth** class, referenced heavily by code — **do not rename in this audit**.

ICDAS 5–6: `dataset/excluded/` (16 files) — out of 0–4 train scope; keep excluded.

## 5. Duplicate / redundant

- Root `STAGE3C_SEED_QC_REPORT.md` / `STAGE3C_SEED_TRAINING_REPORT.md` vs `reports/stage3c_seed_*` and later `reports/TOOTH_DETECTOR_BATCH01_TRAINING.md`.
- Stage 3A CSVs under both `reports/` and `fdi_detection_dataset/reports/`.
- `models/best.keras` ≈ same ordinal contract as `models/deploy.keras`.
- Three local venvs (`.venv`, `.venv-streamlit`, `.venv311`).
- Ultralytics `runs/detect/val*` vs plots already copied to `reports/batch01_yolo_plots/`.

## 6. Generated / temp

- `cropped_teeth/images/` (5676), `overlays/` (420), `manifest.csv`.
- `predictions/` YOLO + ICDAS CSVs.
- `runs/detect/`.
- `reports/batch01_yolo_plots/`, `backend/icdas_predictions.db`, `backend/uploads/`.
- `__pycache__/`, `.pytest_cache/`, `ml/.pytest_cache/`.
- `yolo11n.pt` is a downloaded init weight (keep).

## 7. Suspicious / unknown

- **`ml/.git/`** nested git repo inside `ml/`.
- **`labels.csv/`** empty directory (not a file).
- **`assets/`** directory junction (gitignored).
- **`crm_backend/`** full CRM; pipeline mentions Patient Registration / Visit but **no import** from ICDAS `backend/`. Could be future CRM or unrelated student module — **UNKNOWN**, keep.
- **`models/caries_detector/best.pt`** referenced by `caries_detector.py`; folder not listed under `models/` in this audit — weights may be missing.

## 8. Dependencies (important)

```text
fdi_detection_dataset/images/selected/ (420)
    → tools/run_tooth_cropping.py + ml/src/tooth_cropping.py
        + models/tooth_detector_batch01/weights/best.pt
        → cropped_teeth/

fdi_detection_dataset/tooth_detector_batch01/data.yaml
    → tools/train_tooth_detector_batch01.py + yolo11n.pt
        → models/tooth_detector_batch01/

dataset/annotations.csv  --X-->  dataset/{train,val,test}/0-4/   (BROKEN: 0 pixels)
    → ml/src/dataset.py → ml/train.py → should write 5-class softmax keras
        ≠ models/deploy.keras (actual ordinal 4)

models/deploy.keras
    → ml/src/icdas_predictor.py / backend inference (engine REFUSES 4-out)

backend/app/main.py → inference.py + image_quality.py + groq_service.py
fronted/streamlit_app.py → backend API

docker-compose.yml → models/deploy.keras + ICDAS_NUM_CLASSES=5  (contract vs file mismatch)
```

Lesion path (experimental): `data_external` / `tools/prepare_public_caries_yolo.py` → `caries_detector.py` — **must not** feed ICDAS 0–4 GT.

## 9. Recommended clean architecture (future; not applied)

```text
apps/api/          (today: backend/)
apps/ui/           (today: fronted/)
apps/crm/          (today: crm_backend/ IF it is the registration product)
ml/                (ICDAS train + detector helpers; remove nested .git after review)
data/intraoral_rgb/            (today: fdi_detection_dataset/images/selected)
data/detection_yolo_batch01/   (today: fdi_detection_dataset/tooth_detector_batch01)
data/icdas_labeled/            (today: dataset/ 0-4 pixels — restore first)
data/generated/crops/          (today: cropped_teeth/)
models/detector/               (best.pt)
models/icdas/                  (future softmax 0-4 only)
archive/fdi_historical/
archive/models/icdas_ordinal_and_7class/
archive/experimental/caries_lesion/
```

Keep **tooth** detector and 5676 crops. Do not treat crops as ICDAS GT. Do not overwrite `deploy.keras` until a new 5-class model is evaluated.

## 10. Proposed move plan (do not execute now)

| When | What | Where | Risk |
| --- | --- | --- | --- |
| After review | Root `FDI_*.md`, `FDTooth_*.md`, `RGB_FDI_*.md`, `STAGE2E_RGB_FDI_*`, `reports/*fdi*` | `archive/fdi_historical/` | low |
| After review | `fdi_mapping/` only | `archive/fdi_historical/` | low |
| After backup | `models/icdas_mobilenet_cbam/`, `_ordinal/`, `5class_v2/v3/weighted/` | `archive/models/` | medium |
| After new softmax exists | `deploy.keras` `best.keras` | `archive/models/icdas_ordinal_stale/` | **high** |
| Anytime (low value) | `*.bak_pre_caries_pipeline`, `runs/detect/`, collapsed YOLO run | `archive/` or delete | low |
| After confirm unused | `tools/crop_teeth.py`, lesion train scripts | `archive/experimental/` | medium |
| Never until planned | `fdi_detection_dataset/` rename, `fronted/` rename, `models/tooth_detector_batch01/` | — | **high** |
| Never blindly | `ml/.git`, `assets/` junction, `crm_backend/` | — | **high** |
| Never | 420 originals, Batch 01 labels, `best.pt`, `annotations.csv` | — | **high** |

**ICDAS blocker unchanged:** restore or newly collect dentist-labeled ICDAS 0–4 **pixels** (CSV already has classes 1–3). Then train softmax 5 into a **new** folder; evaluate; only then replace deploy.

---

## Integrity

This audit did not retrain, did not modify datasets or models, and did not change Git history. Only the two report files under `reports/` named in the task were added.
