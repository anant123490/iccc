# Stage 3C-1 — Seed training and candidate labeling

Date: 2026-08-26

**SEED MODEL NOT READY FOR PSEUDO-LABELING**

A separate seed-trained candidate tooth detector was **not** trained. Remaining images were **not** inferred. Candidate folders were **not** filled.

This is a **seed-trained candidate tooth detector** workflow only. Nothing here is clinically validated.

---

## Final decision answers

1. Was Batch_01 successfully verified? **NO** (images present; **0** human boxes in repo).
2. How many human-verified tooth boxes exist? **0**
3. Was a separate tooth detector trained? **NO**
4. What validation metrics were obtained? **n/a** (no train/eval)
5. Was the model considered suitable for candidate generation? **NO**
6. How many remaining images were processed? **0** (of ~360)
7. How many candidate tooth boxes were generated? **0**
8. How many images require human QC? **60** Batch_01 still need **export of drawn boxes** into the repo (plus later QC of any candidates).
9. Were FDI labels generated? **NO**
10. Was ICDAS modified? **NO**
11. Were lesion annotations used as tooth annotations? **NO**
12. Were original RGB images modified? **NO**

---

## Numbers

| Metric | Value |
| --- | --- |
| Seed images (Batch_01 listed) | 60 |
| Seed boxes | 0 |
| Train images | 0 |
| Validation images | 0 |
| Precision / recall / mAP50 / mAP50-95 | n/a |
| Remaining images processed | 0 |
| Candidate boxes | 0 |
| Zero-detection images (candidates) | n/a |
| Low-confidence detections | n/a |
| Suspicious images | n/a |
| Images requiring human QC | 60 seed (missing export); 360 not pseudo-labeled |

Train/val manifests: headers only in `reports/stage3c_seed_train_manifest.csv` and `reports/stage3c_seed_val_manifest.csv` (no split without boxes).

JSON: `reports/stage3c_seed_training_report.json`, `reports/stage3c_candidate_annotation_report.json`.

---

## Protection check

| Resource | Modified |
| --- | --- |
| ICDAS data / labels / models | NO |
| Backend / frontend / Streamlit / FastAPI / Groq | NO |
| Lesion XML | NO |
| Original RGB | NO |
| FDI labels | NO |
| `dataset/`, `ml/` | NO |

New reports only under `reports/` and the two Stage 3C markdown files at repo root.

---

## Next step

**HUMAN:** export Batch_01 from CVAT into `fdi_detection_dataset/annotations/yolo/` (class `0 = tooth` only). Then re-run this stage.

Do not treat empty placeholders as ground truth. Do not start Stage 3D or FDI until seed QC passes and candidates are human-reviewed.
