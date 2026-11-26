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

The dataset reset clears training uploads, detected crops/overlays, labeled copies, confirmed ICDAS dataset exports, and their database records. The full reset also clears Admin training-job records. Patient visits, clinical predictions, model registry records, detector weights, ICDAS model weights, source code, and earlier recovery archives are outside the reset scope. The UI requires the exact confirmation sentence displayed there.

Do not reset if the current images or labels are still needed. The archive is a recovery aid; keep a separate backup according to the project’s data-retention policy.

## Import patient photos

Use **Admin → Training upload** for full-mouth photos. Each batch accepts up to 250 JPG/PNG files (12 MB each, 512 MB total). Select one registered patient for the whole batch, or use a CSV to map each file to its patient. Register a new patient from the upload page or **Patients** first; the database generates its patient ID.

The CSV columns are:

```csv
filename,patient_id,image_id
mouth_001.jpg,<registered-patient-id>,IMG0001
mouth_002.jpg,<registered-patient-id>,IMG0002
mouth_003.jpg,<another-registered-patient-id>,IMG0003
```

`filename` must match each selected upload filename exactly. Every `patient_id` must already exist in the Patient table; the API rejects unregistered IDs and does not invent patient IDs. Never put patient names, phone numbers, or other direct identifiers in the CSV. `image_id` is optional and must be unique when supplied. Multiple images for one patient use the same registered ID.

Uploads receive opaque storage filenames and persist an SHA-256 content hash. Exact duplicates are kept and flagged; unique photos enter a background Tooth Detector V2 job. The upload history shows per-photo and per-batch progress, errors, crop counts, and retry attempts. Failed jobs retain their original photo and can be resumed. Review completed detections in **Box review**: inspect overlays, adjust or delete incorrect boxes, add missed boxes, then verify. Only after verification can crops enter ICDAS labeling.

New photos are saved with `split_name=UNASSIGNED` and are never put into Train, Validation, or Test by the upload step. Assignments are made later, patient-wise, in **Dataset Management**.

## Clinician labeling

In **Labeling**, a clinician assigns ICDAS 0–4 to each verified tooth crop. `UNCERTAIN` and `UNGRADABLE` are available when a defensible grade cannot be recorded. They are retained in the manifest and audit trail but excluded from supervised training. No missing ICDAS grade is inferred from detector output or image appearance.

Changing a box after it has a grade or review status clears that annotation because the crop changed. Box edits, labels, uncertainty, ungradable, crop rejection, identity edits, image deactivation, reset, training completion, and model promotion are recorded in the append-only audit table. Review recent events in **Dataset → Recent annotation and training audit history**. Rejected or skipped data remain excluded from training.

## Assign patients, preview, and export a version

Open **Dataset Management** after box verification and ICDAS labeling. The patient-wise table is loaded from the Admin database and shows original photos, detected crops, labeled and verified crops, ICDAS 0–4 counts, current assignment, and data-quality status. Choose automatic 70/15/15 targets, manual assignment, hybrid assignment, or bulk assignment of selected patients.

The preview reports per-split counts and actual patient percentages, unassigned patients, missing patient IDs, unlabeled and unverified crops, duplicate photos, saved assignment conflicts, and percentage deviations. A warning acknowledgment is required when exclusions or target deviations are present. The safeguard is explicit: only active, unique, box-verified tooth crops with clinician-confirmed ICDAS 0–4 labels are copied into training folders. Missing-ID, unassigned, duplicate, uncertain, ungradable, unlabeled, unverified, and missing-file records remain in the manifests with exclusion reasons.

At least three identified patients and one patient in each split are required. One patient's photos, crops, labels, and verification records inherit the same assignment. Exact duplicate-photo hashes and detector canonical links are grouped before assignment. Confirming a preview writes assignments to the database and creates a new immutable version under `data/icdas/datasets/`; existing versions are never overwritten.

```text
data/icdas/datasets/dataset_v001/
  train/
    images/          # verified tooth crops only
    labels.csv
  validation/
    images/
    labels.csv
  test/
    images/
    labels.csv
  manifest.json
  manifest.csv
  split_assignments.json
  class_mapping.json
  statistics.json
  split_report.md
```

Each split CSV stores the crop filename, ICDAS grade, pseudonymous `patient_id`, `source_photo_id`, `crop_id`, verification status, and hash provenance. Original full-mouth photos are never copied into classifier image folders. Database assignments and version manifests preserve the patient-to-split mapping used for each export.

The existing MobileNetV3 + CBAM pipeline reads the split `labels.csv` files. Its logical validation split maps to the exported `validation/` folder; legacy class-directory datasets remain readable. Integrity checks compare patient, source-photo, original-image hash, and crop assignments across splits; reject duplicate included crops; verify exported image files and labels; and report missing ICDAS classes in train, validation, or test. Backend training requires all five classes in each split, at least five verified training crops, and the configured patient floor.

The planning target is about 100 distinct patients. The default minimum is three real patients so train, validation, and test each have a patient group. Smaller datasets receive reliability warnings; this minimum is not a statistical power calculation or performance guarantee. `ICDAS_MIN_TRAINING_PATIENTS` may raise the floor for a study protocol. Building a dataset does not start training. Upload batch and detector status fields are added through an additive SQLite migration; the backend backs up a database before applying detected schema changes, and existing photo records are preserved.

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
