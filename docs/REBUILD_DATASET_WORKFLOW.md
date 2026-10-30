# Admin dataset rebuild workflow

This guide describes the Admin workflow for rebuilding the ICDAS training dataset. It is separate from the legacy `data/icdas/` folder workflow documented in [DATASET_WORKFLOW.md](DATASET_WORKFLOW.md). Full-mouth photos stay in `data/training/`; they do not enter patient visits or Tooth Detector V2 training.

## Start the app

From PowerShell at the repository root, use the existing shared Python environment:

```powershell
.\scripts\start_backend.ps1
```

In another terminal:

```powershell
.\scripts\start_admin.ps1
```

Open the Admin app at `http://127.0.0.1:8503` and sign in with `ICDAS_ADMIN_PASSWORD` from `app/backend/.env`.

## Optional reset

Reset is an explicit Admin action and is never run by startup, dataset building, or training. First inspect **Dashboard → Dataset Management → Reset plan**. It reports affected files and database row counts. A reset creates a ZIP recovery archive under `data/training/backups/` before deleting anything and writes a persistent `training_audit_events` entry.

The dataset reset clears training uploads, detected crops/overlays, labeled copies, dataset versions, and their database records. The full reset also clears Admin training-job records. Patient visits, clinical predictions, model registry records, detector weights, ICDAS model weights, source code, and earlier recovery archives are outside the reset scope. The UI requires the exact confirmation sentence displayed there.

Do not reset if the current images or labels are still needed. The archive is a recovery aid; keep a separate backup according to the project’s data-retention policy.

## Import patient photos

Use **Training upload** for the new full-mouth photos. A CSV manifest is recommended for batches. The accepted columns are:

```csv
filename,patient_id,image_id
mouth_001.jpg,PAT001,IMG0001
mouth_002.jpg,PAT001,IMG0002
mouth_003.jpg,PAT002,IMG0003
```

`filename` must match each selected upload filename exactly. `patient_id` must be a pseudonymous code such as `PAT001`; do not put patient names, phone numbers, or other direct identifiers in the CSV. `image_id` is optional and must be unique when supplied. Multiple images for one patient should use the same `patient_id` and different `image_id` values. If a CSV is omitted, assign patient IDs in **Box review** before building a dataset.

Uploads receive opaque storage filenames. Exact and near-duplicate images remain visible for review but are excluded from training. The existing Tooth Detector V2 proposes boxes for unique images. In **Box review**, inspect the overlay, adjust or delete incorrect boxes, add missed boxes, then verify. A verified box set is required before labels enter the dataset.

## Clinician labeling

In **Labeling**, a clinician assigns ICDAS 0–4 to each verified tooth crop. `UNCERTAIN` and `UNGRADABLE` are available when a defensible grade cannot be recorded. They are retained in the manifest and audit trail but excluded from supervised training. No missing ICDAS grade is inferred from detector output or image appearance.

Changing a box after it has a grade or review status clears that annotation because the crop changed. Box edits, labels, uncertainty, ungradable, crop rejection, identity edits, image deactivation, reset, training completion, and model promotion are recorded in the append-only audit table. Review recent events in **Dataset → Recent annotation and training audit history**. Rejected or skipped data remain excluded from training.

## Build and validate a version

The Admin dashboard reports patient, image, crop, annotation, duplicate, and review counts. Dataset building stays disabled until every unique active photo has a pseudonymous patient ID, the boxes are reviewed, and the active verified crops are labeled, marked uncertain, or marked ungradable. At least three distinct patients are needed to populate train, validation, and test splits.

Build creates an immutable `vN` directory under `data/training/versions/`. It uses a deterministic seed (`42`) and a 70/15/15 target split by **patient**. All images and crops belonging to one patient stay together. Each version saves:

- `manifest.csv` with patient/image/crop provenance and inclusion or exclusion reasons
- `split_assignments.json` with patient and image assignments
- `class_mapping.json` for ICDAS 0–4
- `statistics.json` with split counts and validation results
- the `train/`, `val/`, and `test/` class folders (`valid/` is a compatibility copy of `val/`)

Validation checks patient, image, source, crop, and content-hash leakage; duplicate crops; nonempty splits; ICDAS classes 0–4 in training; and support for all five classes in the held-out test split. A version with validation issues can be inspected, but backend training rejects it.

The planning target is about 100 distinct patients. The default minimum training gate is 90 and can be changed with `ICDAS_MIN_TRAINING_PATIENTS` in the backend environment; it must be at least three so all three splits can be populated. The default of 90 preserves the original plan while allowing a modest shortfall from the approximate 100-patient target. It is an operational floor, not a statistical power calculation, evidence that 90 patients are present, or a guarantee of clinical performance. Choose a different value only from the study protocol and intended claims, and document that decision. Lowering the floor does not relax patient-level split isolation, class-support, duplicate-conflict, or manifest validation checks. Training remains blocked below the configured floor, when any class is absent from train or test, when duplicate-crop conflicts exist, or when validation fails. Building a dataset does not start training.

## Train, evaluate, and promote

Training is off by default. Once the dataset and labels have passed clinician review, configure `ICDAS_ALLOW_TRAIN=1` in the backend environment and restart the backend. **Admin → Training** checks the gates again before launching the existing MobileNetV3Small + CBAM training pipeline; Tooth Detector V2 is not retrained.

Each candidate model is written to `models/icdas/vN/`. The model directory stores the exact training config, dataset manifest, class mapping, split assignments, dataset statistics, and a provenance file with artifact checksums. The trainer saves the best and final Keras models, epoch history, and held-out test results including accuracy, macro/weighted F1, per-class metrics, and a confusion matrix. A successful training run is registered as an inactive model.

Review the candidate’s held-out metrics in **Model registry** before choosing **SET ACTIVE**. Promotion records an audit event and changes the local inference pointer only; it does not publish a cloud model. The backend rejects promotion if the held-out results are missing a metric, a confusion matrix, or support for any ICDAS class. Do not describe research/screening metrics as clinical accuracy.

## Tests

From `app/backend`:

```powershell
pytest -q tests/test_admin_reset.py tests/test_dataset_hardening.py tests/test_crop_duplicates.py tests/test_photo_deactivation.py
```

Tests use temporary SQLite databases and training directories. They do not reset the project database, start model training, or modify production model weights.
