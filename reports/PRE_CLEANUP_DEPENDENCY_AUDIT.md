# Pre-cleanup dependency audit

**Date:** 2026-08-27  
**Scope:** Safety check of the move plan in `reports/REPOSITORY_ORGANIZATION_AUDIT.md`.  
**Constraint:** This document only. No files were moved, deleted, renamed, or otherwise modified except this report.

**Intended product:** RGB intraoral photograph → image quality → whole-tooth detection → individual tooth crops → ICDAS 0–4 softmax → Grad-CAM → clinical report. **FDI is out of scope.**

**ICDAS contract (intended):** 5 classes, ICDAS 0–4, softmax, `ordinal_regression: false`.  
**On-disk `models/deploy.keras`:** stale 4-output ordinal checkpoint. **Do not replace it.**  
**643 ICDAS CSV rows:** pixels not on disk. **5,676 detector crops are not ICDAS ground truth.** **Do not retrain ICDAS.**

---

## How this audit was done

Repository search (not filename-only) across Python, YAML, JSON, Markdown, Docker Compose, and scripts for:

- `Path(...)`, `os.path.join`, `../`, `./`, `glob(`, `Path(__file__)`
- hardcoded folder names (`fdi_detection_dataset`, `fronted`, `crm_backend`, `cropped_teeth`, `runs/detect`, `labels/labels.csv`, model paths)
- imports (`caries_pipeline`, `crop_teeth`, `tooth_cropping`)

Filesystem checks: nested `ml/.git`, `assets/` junction vs symlink vs ordinary directory, empty `labels.csv/` directory.

---

## Verdict summary

**Almost nothing in the previous move plan is SAFE TO MOVE with zero follow-up.**  
Runtime and dataset paths are hardcoded. Several “historical” items are still imported or read.

| Recommendation | Meaning |
| --- | --- |
| **SAFE TO MOVE** | No runtime, import, config, dataset, or script dependency found. Docs-only links are listed; moving still leaves stale docs unless those are updated. |
| **MOVE ONLY AFTER UPDATING REFERENCES** | Dependencies exist; a move is possible only after a coordinated path update. **Do not do that now.** |
| **LEAVE IN PLACE** | Current pipeline, data, or API would break, or the item is protected. |
| **MANUAL REVIEW REQUIRED** | Nested git, junction, empty-dir accident, or product-ownership unknown. |

---

## 1. Safe-to-move items

None of the **protected assets** (420 RGB originals, Batch 01 dataset, `best.pt`, 5,676 crops, ICDAS source, `annotations.csv`, FastAPI, Streamlit) are safe to move.

Items that are **closest** to low-risk archival (still **do not move in this audit**; they are generated/historical and nothing **loads** them at runtime):

| Current path | Proposed destination (prior audit) | Why relatively low runtime risk | Remaining caveat |
| --- | --- | --- | --- |
| `models/tooth_detector_batch01_run1_adamw_lr0.01_collapsed/` | `archive/` | No Python/YAML loader found | Training reports mention it as history |
| `models/archives/tooth_detector_batch01_*` | `archive/models/yolo_runs/` | Archive copies; live weights are `models/tooth_detector_batch01/weights/` | Keep until you confirm no one points at a specific timestamp folder |
| `models/icdas_mobilenet_cbam_5class_v2/`, `_v3/`, `_weighted/` | `archive/models/` | **No** `.py` / `.yaml` path strings to these folder names | README glob `models/icdas_mobilenet_cbam*` would still match if they stay; docs say “do not overwrite” |

**Recommendation for the table above:** treat as **MOVE ONLY AFTER UPDATING REFERENCES** for documentation, or **LEAVE IN PLACE** until an explicit cleanup pass. **Not SAFE TO MOVE** under a conservative bar, because training history and README globs still name the parent pattern.

**Root FDI markdown** (`FDI_RGB_DATASET_SEARCH.md`, `FDI_RGB_FINAL_VERIFICATION.md`, `FDTooth_ACQUISITION_REPORT.md`, `RGB_FDI_DATASET_SEARCH_STAGE2D6.md`, `STAGE2E_RGB_FDI_FEASIBILITY_REPORT.md`): **no Python/YAML references.** One doc still cites them: `reports/stage3c_roboflow_test/sources.md`.  
**Recommendation:** **MOVE ONLY AFTER UPDATING REFERENCES** (that markdown plus any README links), not SAFE TO MOVE.

**`reports/*fdi*` and `reports/stage2d*_fdi_*`:** documentation-only. Same recommendation.

---

## 2. Unsafe-to-move items

These must **LEAVE IN PLACE**. Moving or renaming them would break the current camera / detection / crop / ICDAS-code paths.

| Current path | Proposed destination | Risk | Why unsafe |
| --- | --- | --- | --- |
| `fdi_detection_dataset/` (entire tree, especially `images/selected/` and `tooth_detector_batch01/`) | `data/intraoral_rgb/` | **HIGH** | Hardcoded in crop pipeline, YOLO train/verify/report, `data.yaml` `path:`, `args.yaml`, Batch `image_list.json` `relative_path`, CVAT/stage docs |
| `fdi_detection_dataset/images/selected/` (420 originals) | keep | **HIGH** | `ml/src/tooth_cropping.py` `DEFAULT_SOURCE` |
| `fdi_detection_dataset/tooth_detector_batch01/` (767 boxes, 46/6/8) | keep | **HIGH** | `tools/train_tooth_detector_batch01.py`, `verify_tooth_detector_batch01.py`, `report_tooth_detector_train.py`, Ultralytics `data.yaml` |
| `models/tooth_detector_batch01/weights/best.pt` | keep | **HIGH** | `DEFAULT_WEIGHTS` in `ml/src/tooth_cropping.py`; live detector |
| `models/tooth_detector_batch01/` (live run dir) | keep | **HIGH** | Train script `OUT`; `args.yaml` `save_dir` |
| `models/deploy.keras`, `models/best.keras` | archive after new softmax | **HIGH** | `backend/app/config.py`, `ml/src/icdas_predictor.py`, `ml/export.py`, `docker-compose.yml` (`ICDAS_DEPLOY_MODEL_PATH`, `ICDAS_MODEL_PATH`). **Do not replace `deploy.keras`.** |
| `dataset/`, `dataset/annotations.csv` | keep | **HIGH** | `ml/train.py` `dataset_root`; 643 rows; empty pixel dirs still expected |
| `cropped_teeth/` | `data/generated/cropped_teeth/` | **HIGH** | Defaults in `tooth_cropping.py`, `run_icdas_batch_prediction.py`, `label_icdas.py`, `build_dataset.py`, `tools/common.py` |
| `yolo11n.pt` (repo root) | `models/pretrained/` | **HIGH** | Train init; `models/tooth_detector_batch01/args.yaml` stores absolute path to this file |
| `fronted/` | `apps/ui/` or `frontend/` | **HIGH** | README, `docs/SETUP.md`, `docs/DEPLOYMENT.md`, `docs/ARCHITECTURE.md`, Streamlit command `streamlit run fronted/streamlit_app.py` |
| `backend/` | `apps/api/` | **HIGH** | Docker, CI, Streamlit API base URL, tests |
| `ml/` (except nested `.git` question) | keep | **HIGH** | Train/export/predictor; `Path(__file__).parents[1]` / `parents[2]` assume repo layout |
| `annotation_batches/`, `annotation_project/` | keep | **HIGH** | JSON `relative_path` into `fdi_detection_dataset/images/selected/` |
| `backend/app/caries_pipeline.py`, `caries_detector.py` | `archive/experimental/` | **HIGH** | **Live FastAPI:** `backend/app/main.py` imports and calls `run_localized_pipeline` on `/api/v1` analyze |

---

## 3. Items requiring reference updates

If a future cleanup ever proceeds, these cannot move until every listed reference is updated **in the same change**.

### 3.1 `fdi_mapping/` → `archive/fdi_historical/`

| Field | Value |
| --- | --- |
| Current path | `fdi_detection_dataset/annotations/fdi_mapping/` |
| Proposed | `archive/fdi_historical/` |
| Python | `fdi_detection_dataset/metadata/_build_stage3a.py`: `ANN_FDI = OUT / "annotations" / "fdi_mapping"` (creates/expects that directory under the dataset tree) |
| Imports | **None** (no `import` of mapping tables) |
| Config | None |
| Dataset | Nested under `fdi_detection_dataset/`; moving it without the parent is a partial tree split |
| Docs | `README_STAGE3B.md`, `fdi_detection_dataset/README.md`, `STAGE3A_DETECTION_DATASET_REPORT.md` |
| Break if moved? | Re-running Stage 3A builder would recreate or fail depending on mkdir vs read. Runtime inference **does not** load this folder. |
| Recommendation | **MOVE ONLY AFTER UPDATING REFERENCES** (script + docs). Prefer **LEAVE IN PLACE** until `fdi_detection_dataset/` itself is planned. |

### 3.2 7-class / ordinal ICDAS experiment folders

| Current path | Proposed | References | Recommendation |
| --- | --- | --- | --- |
| `models/icdas_mobilenet_cbam/` | `archive/models/` | `tools/audit_icdas_classifier.py` **reads** `models/icdas_mobilenet_cbam/config.json`; `docs/TRAINING.md`; `docs/PROJECT_REPORT.md` `file://` links to `test_evaluation/` | **MOVE ONLY AFTER UPDATING REFERENCES** |
| `models/icdas_mobilenet_cbam_ordinal/` | `archive/models/` | Pattern `icdas_mobilenet_cbam*` in README, `ml/configs/icdas_v2.yaml` comment, `tools/evaluate_icdas.py` | **MOVE ONLY AFTER UPDATING REFERENCES** |
| `ml/configs/default.yaml` `experiment_name` / `icdas_mobilenet_cbam_5class_v4` | n/a | `ml/train.py` writes `models/{experiment_name}/` | **LEAVE IN PLACE** (active train layout, not a move target) |

**Do not confuse these folders with `deploy.keras`.** Isolating experiment trees later is correct; deleting them is not.

### 3.3 `cropped_teeth/` → `data/generated/...`

Scripts that must change together: `ml/src/tooth_cropping.py` (`DEFAULT_OUT`), `tools/run_tooth_cropping.py`, `tools/run_icdas_batch_prediction.py`, `tools/label_icdas.py`, `tools/build_dataset.py`, `tools/common.py`, `tools/crop_teeth.py` `--out` default, plus reports (`TOOTH_CROPPING_REPORT.md`).

**Recommendation:** **MOVE ONLY AFTER UPDATING REFERENCES**. Crops are an important asset (5,676 files) but **not** ICDAS GT.

### 3.4 `data_external/detection/` → `archive/external/zenodo_lesion/`

Hardcoded in `tools/prepare_public_caries_yolo.py`, `tools/train_caries_detector.py`, `tools/icdas_v2_lib.py`, `fdi_detection_dataset/metadata/_build_stage3a.py`.

**Recommendation:** **MOVE ONLY AFTER UPDATING REFERENCES**. Lesion `d`/`D` data must not become ICDAS GT; it is still used by experimental lesion YOLO **and** Stage 3A builder source paths.

### 3.5 `predictions/` YOLO + ICDAS CSVs

Code defaults and reports cite `predictions/labels|visualizations|confidence_reports` and `predictions/icdas_predictions/`. Train script explicitly **must not** use `predictions/` as labels.

**Recommendation:** **MOVE ONLY AFTER UPDATING REFERENCES** (docs/tools that glob these paths). Runtime FastAPI does not load these CSVs.

### 3.6 `tools/crop_teeth.py` → archive

Referenced by `tools/selftest.py` (invokes `--help` and a crop run), `tools/label_icdas.py` user message, `tools/README.md`. Current production crop path is `ml/src/tooth_cropping.py` + `tools/run_tooth_cropping.py`.

**Recommendation:** **MOVE ONLY AFTER UPDATING REFERENCES** (or **LEAVE IN PLACE**). Not unused.

### 3.7 Root FDI docs / `reports/*fdi*`

See §1. Update `reports/stage3c_roboflow_test/sources.md` and any README lists first.

---

## 4. Items requiring manual review

| Item | Issue | Recommendation |
| --- | --- | --- |
| `ml/.git/` | Nested Git repository (see §6) | **MANUAL REVIEW REQUIRED** / **LEAVE IN PLACE** |
| `assets/` | Directory **junction** (see §7) | **LEAVE IN PLACE** / **MANUAL REVIEW REQUIRED** |
| `crm_backend/` | Separate FastAPI CRM; not imported by camera API, but product ownership unknown (see special item 5) | **LEAVE IN PLACE** / **MANUAL REVIEW REQUIRED** |
| `labels.csv/` (empty **directory**) | Name collides with a CSV; code expects `labels/labels.csv` **file** (see special item 10) | **MANUAL REVIEW REQUIRED** |
| `models/caries_detector/best.pt` | `caries_detector.py` loads this path; weights may be **missing** (code logs warning and returns empty detections) | **MANUAL REVIEW REQUIRED** — do not delete the Python modules while FastAPI still imports them |
| `*.bak_pre_caries_pipeline` | Unique pre-pipeline snapshots; nothing imports them (see special item 7) | **LEAVE IN PLACE** until a human diffs them |
| `runs/detect/` | Ultralytics default dump dir; current code does not load `val*` for inference (see special item 6) | **LEAVE IN PLACE** |
| `STAGE3C_SEED_*.md` (root) vs `reports/` copies | Duplicate docs | **MOVE ONLY AFTER UPDATING REFERENCES** if any README still points at root |

---

## Per-item table (every proposed move from the prior plan)

| Current path | Proposed destination | Files that reference it | Import / config / dataset / script / docs | Break if moved? | Refs need update? | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Root `FDI_*.md`, `FDTooth_*.md`, `RGB_FDI_*.md`, `STAGE2E_RGB_FDI_*` | `archive/fdi_historical/` | `reports/stage3c_roboflow_test/sources.md`; prior org audit | Docs only (no `.py`) | Docs links only | Yes | LOW | **MOVE ONLY AFTER UPDATING REFERENCES** |
| `reports/*fdi*`, `reports/stage2d*_fdi_*` | `archive/fdi_historical/` | Cross-links among reports | Docs | Docs | Yes | LOW | **MOVE ONLY AFTER UPDATING REFERENCES** |
| `fdi_mapping/` | `archive/fdi_historical/` | `_build_stage3a.py`, Stage 3A/3B READMEs | Script mkdir + docs; **no** runtime import | Stage 3A rebuild / docs | Yes | MEDIUM | **MOVE ONLY AFTER UPDATING REFERENCES**; prefer leave nested |
| `fdi_detection_dataset/` rename | `data/intraoral_rgb/` | Many `.py`, `data.yaml`, `args.yaml`, `annotation_batches/*/image_list.json`, crop `run_summary.json` | Dataset + train + crop | **Yes** | Yes | **HIGH** | **LEAVE IN PLACE** |
| `fronted/` rename | `apps/ui/` | README, SETUP, DEPLOYMENT, ARCHITECTURE | Docs + operator commands | Streamlit launch | Yes | **HIGH** | **LEAVE IN PLACE** |
| YOLO `best.pt` / Batch 01 / 420 JPGs / `annotations.csv` | keep / never | Train, crop, `ml/train.py` | Core assets | **Yes** | N/A | **HIGH** | **LEAVE IN PLACE** |
| `deploy.keras` / `best.keras` | `archive/models/icdas_ordinal_stale/` **after** new softmax | `config.py`, `icdas_predictor.py`, `audit_icdas_classifier.py`, `export.py`, `docker-compose.yml` | Model paths | **Yes** | Yes | **HIGH** | **LEAVE IN PLACE** (do not replace) |
| `ml/.git` | never blindly | Nested repo only | Git | Detaches `ml` history | N/A | **HIGH** | **MANUAL REVIEW REQUIRED** |
| `assets/` | never | `scripts/import_whatsapp_images.py`; `.gitignore` `assets/` | Junction + ingest script | Ingest / Cursor assets | Do not change junction | **HIGH** | **LEAVE IN PLACE** |
| `crm_backend/` | later `apps/crm/` | Own README only vs ICDAS app | Standalone app | Unknown product | Unknown | MEDIUM–HIGH | **LEAVE IN PLACE** |
| `models/icdas_mobilenet_cbam*` experiment dirs | `archive/models/` | `audit_icdas_classifier.py`, TRAINING.md, PROJECT_REPORT.md, icdas_v2 comments | Config JSON + docs | Audit script + docs | Yes | MEDIUM | **MOVE ONLY AFTER UPDATING REFERENCES** |
| `runs/detect/` | `archive/` or delete | `reports/stage3c_github_detector_audit/` cites **historical** `runs/detect/train4/weights/best.pt` (not the current val dumps) | Docs; Ultralytics recreates `runs/` on `val()` | Next val just recreates; do not delete if you rely on those plots | Docs optional | LOW | **LEAVE IN PLACE** |
| `*.bak_pre_caries_pipeline` | `archive/` | None import `.bak` | Unique backup content | Lose unique code if deleted | Diff first | LOW–MEDIUM | **LEAVE IN PLACE** |
| `tools/crop_teeth.py` | `archive/experimental/` | `selftest.py`, `label_icdas.py`, `tools/README.md` | Scripts | Selftest / operator messages | Yes | MEDIUM | **LEAVE IN PLACE** |
| Lesion train scripts + `caries_*.py` | `archive/experimental/` | `main.py` **imports** pipeline; tests; `prepare_public_caries_yolo.py`; `train_caries_detector.py` | **Active API path** | **Yes — FastAPI analyze** | Yes | **HIGH** | **LEAVE IN PLACE** |
| `data_external/detection/` | `archive/external/` | prepare/train caries, `icdas_v2_lib.py`, `_build_stage3a.py` | Dataset paths | Those tools | Yes | MEDIUM | **MOVE ONLY AFTER UPDATING REFERENCES** |
| Collapsed YOLO run / `models/archives/` | `archive/` | Training reports | Historical | Reports only | Optional | LOW | **LEAVE IN PLACE** until explicit archive |
| `labels.csv/` empty dir | manual | **No** code uses this directory | Accidental vs `labels/labels.csv` | Unlikely | Clarify | LOW | **MANUAL REVIEW REQUIRED** |
| `labels/` | merge later | `label_icdas.py`, `build_dataset.py`, `check_dataset.py`, `tools/README.md` | Expected **file** `labels/labels.csv` | Labeling tools | Yes if renamed | MEDIUM | **LEAVE IN PLACE** |
| Nested venvs / pytest cache | gitignored | Local only | — | Local env | No | LOW | Do not move as “source”; ignore |

---

## 5. Exact dependencies discovered

### 5.1 Detection / 420 RGB / 767 boxes / `best.pt`

```
fdi_detection_dataset/images/selected/     (420 originals)
    DEFAULT_SOURCE  ←  ml/src/tooth_cropping.py  (Path(__file__).parents[2])
    relative_path   ←  annotation_batches/*/image_list.json

fdi_detection_dataset/tooth_detector_batch01/data.yaml
    path: <absolute>/fdi_detection_dataset/tooth_detector_batch01
    ← tools/train_tooth_detector_batch01.py
    ← tools/verify_tooth_detector_batch01.py
    ← tools/report_tooth_detector_train.py
    ← models/tooth_detector_batch01/args.yaml  (data: …\data.yaml)

models/tooth_detector_batch01/weights/best.pt
    DEFAULT_WEIGHTS  ←  ml/src/tooth_cropping.py
    OUT              ←  tools/train_tooth_detector_batch01.py

yolo11n.pt  (repo root)
    Ultralytics init; args.yaml model: …\yolo11n.pt
```

### 5.2 Crops (5,676) — not ICDAS GT

```
cropped_teeth/          DEFAULT_OUT  ←  ml/src/tooth_cropping.py
cropped_teeth/images/   CROPS        ←  tools/run_icdas_batch_prediction.py
                        glob         ←  tools/audit_icdas_classifier.py
                        label_icdas.py, build_dataset.py defaults
```

### 5.3 ICDAS code vs stale checkpoint

```
ml/configs/default.yaml     ordinal_regression: false, 5-class softmax (intended)
ml/train.py                 dataset_root default "dataset"; output models/{experiment_name}/
                            overwrite_root_checkpoints → models/deploy.keras (do not run)
models/deploy.keras         4-output ordinal  ←  icdas_predictor.py, backend/app/config.py
docker-compose.yml          ICDAS_MODEL_PATH=/app/models/best.keras
                            ICDAS_DEPLOY_MODEL_PATH=/app/models/deploy.keras
dataset/annotations.csv     643 rows; dataset/{train,val,test}/0–4 have 0 pixels
```

### 5.4 Camera API (FastAPI + Streamlit)

```
fronted/streamlit_app.py  →  backend HTTP API
backend/app/main.py
    from .caries_pipeline import run_localized_pipeline   ← LIVE analyze path
backend/app/caries_pipeline.py
    from .caries_detector import get_caries_detector
    DEFAULT_WEIGHTS = models/caries_detector/best.pt      ← may be missing
backend/app/inference.py  →  deploy.keras (refuses 4-output for softmax engine)
```

Streamlit does **not** import `caries_pipeline` directly; it talks to the API. Moving `fronted/` still breaks documented launch commands.

### 5.5 `fdi_mapping/`

Only `_build_stage3a.py` + Markdown. No classifier/detector import.

### 5.6 `crm_backend/`

No imports from `backend/` or `fronted/`. Self-contained (`cd crm_backend`). Pipeline docs mention Patient Registration conceptually; **not wired**.

### 5.7 Relative / `__file__` layout assumptions

| File | Root resolution |
| --- | --- |
| `ml/src/tooth_cropping.py` | `parents[2]` = repo root |
| `backend/app/config.py` | `BACKEND_DIR.parent` = repo root |
| `backend/app/caries_detector.py` | `parents[2]` = repo root |
| `tools/*.py` | `parents[1]` = repo root |
| `ml/train.py` | `parents[1]` then `.parent` = repo root |
| `scripts/import_whatsapp_images.py` | `parents[1]` + `assets/` |

Moving `ml/` or `backend/` without updating these **breaks** path resolution.

---

## 6. Nested Git repository analysis (`ml/.git/`)

| Check | Result |
| --- | --- |
| Nested `.git` present? | **Yes** — `ml/.git/` |
| Parent repo | `C:/Users/anant/OneDrive/Desktop/icdas project` (remote `iccc.git`) |
| Nested remote | `https://github.com/anant123490/protoype-icdas.git` |
| Nested branch / tip | `main`, first commit `30ee2e9` (“first commit”) |
| Parent `.gitmodules` | **None** — not a declared submodule |
| Intent | **Likely accidental** leftover from an older prototype clone (`protoype-icdas` spelling), **or** an intentional private history for `ml/` that was never linked as a submodule |

**Do not move or delete `ml/.git`.** Effects of touching it:

- Parent Git may already track `ml/` files as normal files **or** treat `ml` as an embedded repo depending on how it was added.
- Deleting nested `.git` does not delete `ml/` source, but **destroys** that nested history.
- Moving `ml/` with nested `.git` moves two repositories at once.

**Recommendation:** **MANUAL REVIEW REQUIRED**. Inspect `git ls-files ml` in the parent vs `git -C ml status` before any cleanup. Leave both repos where they are.

---

## 7. Junction / symlink analysis (`assets/`)

| Check | Result |
| --- | --- |
| Exists | Yes |
| Ordinary directory? | No |
| Symlink (`os.path.islink`)? | **False** |
| Windows reparse | **Junction** (`Attributes`: Directory, ReparsePoint; `LinkType`: Junction) |
| Target | `C:\Users\anant\.cursor\projects\c-Users-anant-OneDrive-Desktop-icdas-project\assets` |
| Contents (workspace listing) | Empty (0 files) — target may also be empty |
| Git | `.gitignore` includes `assets/` |
| Code | `scripts/import_whatsapp_images.py` prefers `PROJECT_ROOT/assets` as “Cursor-uploaded image junction”; fallback path is the same Cursor assets directory; error text documents `mklink /J` |

**Do not modify the junction.** Moving `assets/` would break or duplicate Cursor’s upload mount.

---

## 8. FDI dependency analysis

**Product:** FDI numbering is **out of scope**.

**Folder name `fdi_detection_dataset/` is historical.** Contents used today:

- RGB intraoral photographs (`images/selected/`)
- Whole-tooth YOLO dataset (`tooth_detector_batch01/`, class `0 = tooth`)
- Placeholder YOLO txts under `annotations/yolo/` (Batch 01 **real** labels live under `tooth_detector_batch01/labels`)

**Code depends on the path string `fdi_detection_dataset`, not on FDI labels.**

| Path | Still depended on? | Action |
| --- | --- | --- |
| `fdi_detection_dataset/` | **Yes** — crop, train, verify, annotation JSON | **LEAVE IN PLACE** |
| `fdi_detection_dataset/annotations/fdi_mapping/` | Stage 3A **builder path** + docs only; no inference import | Do not move until builder + docs updated; **not** required for ICDAS 0–4 |
| Root FDI search markdown | Docs only | Archive later after link updates |
| `fdi_mapping` as a Python package | **No** | — |

**Do not rename `fdi_detection_dataset/` until every `relative_path`, YAML `path:`, and `Path(...)` join is updated in one planned change.** The previous audit’s “high risk” rating is confirmed.

---

## Special caution items (requested)

### 1. `fdi_detection_dataset/`

**LEAVE IN PLACE.** Proven dependencies: `ml/src/tooth_cropping.py`, train/verify/report tools, Ultralytics YAML, annotation batch JSON, `cropped_teeth/run_summary.json`.

### 2. `fdi_mapping/`

Not imported by ICDAS or FastAPI. Still referenced by `_build_stage3a.py` and Stage 3A/3B markdown. **MOVE ONLY AFTER UPDATING REFERENCES**; safest is leave nested.

### 3. `ml/.git/`

Nested Git, different remote than parent, not a submodule. **MANUAL REVIEW REQUIRED. Do not move or delete.**

### 4. `assets/`

**Junction** to Cursor project assets. **LEAVE IN PLACE. Do not modify.**

### 5. `crm_backend/`

Not imported or executed by `backend/` camera API or Streamlit. Standalone CRM. **Do not assume unused for the product** (registration/visit may be intended later). **LEAVE IN PLACE.**

### 6. Ultralytics `runs/`

Current `runs/detect/` is val plot dumps. Inference/train use `models/tooth_detector_batch01/`, not `runs/detect/val*`. Historical audit mentions `runs/detect/train4/weights/best.pt` which is **not** the Batch 01 detector. **LEAVE IN PLACE**; do not delete.

### 7. `.bak` files

| Backup | Current counterpart |
| --- | --- |
| `backend/app/main.py.bak_pre_caries_pipeline` | `backend/app/main.py` |
| `backend/app/schemas.py.bak_pre_caries_pipeline` | `backend/app/schemas.py` |
| `backend/app/groq_service.py.bak_pre_caries_pipeline` | `backend/app/groq_service.py` |
| `fronted/streamlit_app.py.bak_pre_caries_pipeline` | `fronted/streamlit_app.py` |

Nothing imports `*.bak*`. They are **pre–caries-pipeline snapshots** and may contain unique code not in the current files. **Do not delete.** **LEAVE IN PLACE.** Diff before any archive.

### 8. 7-class / ordinal ICDAS model folders

Historical, but `tools/audit_icdas_classifier.py` still **reads** `models/icdas_mobilenet_cbam/config.json`. Docs point at `test_evaluation/`. Isolate later; **do not delete**. **MOVE ONLY AFTER UPDATING REFERENCES.** `deploy.keras` stays.

### 9. Lesion-caries experiment

**Currently active in FastAPI** (`run_localized_pipeline`). Tools still point at `data_external/detection/`. **LEAVE IN PLACE.** Moving modules without changing `main.py` **breaks** the camera analyze endpoint. Whole-tooth YOLO is the intended detector; the API has **not** been rewired to it.

### 10. Empty `labels.csv/` directory

- **Exists:** directory `labels.csv/` (empty). **Not** a CSV file.
- **Code expects:** file `labels/labels.csv` (`tools/label_icdas.py` `project_path("labels", "labels.csv")`, `build_dataset.py`, `check_dataset.py`, `tools/README.md`).
- **`labels/`** exists with `.gitkeep` only. `DATASET_REPORT.md` notes `labels/labels.csv` **absent**.
- **Why `labels.csv/` exists:** not created by those tools (they write under `labels/`). Likely accidental (`mkdir labels.csv` instead of a file). **No code expects this exact directory path.**
- **Do not delete `labels/`.** Do not assume `labels.csv/` is the labeling store. **MANUAL REVIEW REQUIRED** (empty dir only).

---

## 9. Recommended cleanup order (when cleanup is explicitly approved — **not now**)

1. **Do not start** until ICDAS pixels exist and a new 5-class softmax is validated. Cleanup does not unblock ICDAS.
2. Human decision on **`ml/.git`**, **`assets/` junction**, **`crm_backend/`** (keep vs later `apps/crm`).
3. Diff **`.bak`** files; archive only after confirming no unique logic.
4. Update **docs only** for FDI search markdown → `archive/fdi_historical/` (lowest runtime risk).
5. Update **`audit_icdas_classifier.py` + TRAINING/PROJECT_REPORT** then isolate 7-class/ordinal **experiment directories** (not `deploy.keras`).
6. **Rewire FastAPI** from lesion `caries_pipeline` to whole-tooth crop + ICDAS **before** moving lesion scripts or `data_external`.
7. Only after a dedicated path-migration PR: `cropped_teeth/`, `predictions/`, `data_external/`.
8. **Last / optional:** rename `fdi_detection_dataset/` and `fronted/` — highest blast radius (JSON relative paths, YAML absolute `path:`, operator commands).

Never in this order: delete junctions, nested git, `best.pt`, 420 originals, 767 boxes, `annotations.csv`, or replace `deploy.keras`.

---

## 10. Final DO NOT TOUCH list

Do not move, delete, rename, or overwrite:

1. `fdi_detection_dataset/images/selected/` (420 original RGB intraoral photographs)
2. `fdi_detection_dataset/tooth_detector_batch01/` (767 verified tooth boxes, patient-aware split)
3. `models/tooth_detector_batch01/weights/best.pt` (and live `last.pt` / run dir)
4. `cropped_teeth/` (5,676 generated crops — not ICDAS GT; still a current asset)
5. ICDAS training source under `ml/` (`train.py`, `src/`, `configs/default.yaml`)
6. `dataset/annotations.csv` (643 rows) and `dataset/{train,val,test}/` layout
7. `models/deploy.keras` and `models/best.keras` (**stale ordinal; do not replace**)
8. `backend/` FastAPI application
9. `fronted/` Streamlit application
10. `yolo11n.pt` until train scripts and `args.yaml` are updated together
11. `ml/.git/` (nested repository)
12. `assets/` junction
13. `crm_backend/` until product ownership is decided
14. `*.bak_pre_caries_pipeline` until human diff
15. Git history

---

## Integrity

This audit did not retrain, did not modify datasets, models, configuration, or source (except adding this file). It did not change Git history.

**NO FILES WERE MOVED, DELETED, RENAMED, OR MODIFIED.**
