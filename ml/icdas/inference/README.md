`models/icdas/current/deploy.keras` is the only production ICDAS path (5-class softmax).
Historical stale ordinal files must not be loaded.

Helpers: `tools/run_icdas_inference.py`, `tools/run_icdas_batch_prediction.py` (do not treat outputs as GT).
