# STALE — removed from production inference

This folder held historical **4-output ordinal** Keras checkpoints (`deploy.keras`, `best.keras`).

Those weight files are **not** a 5-class softmax ICDAS classifier and must **never** be loaded by the patient/admin API.

The keras files were removed from this folder after they were disconnected from production startup. Training code, datasets, labeling, Grad-CAM, and model-registry code were left in place.

Approved production weights belong at:

`models/icdas/current/deploy.keras`

See `reports/ICDAS_CLASSIFIER_AUDIT.md`.
