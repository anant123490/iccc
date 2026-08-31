# Annotation Batch 02

Images: **60** (YOLO-informed selection from the 360 remaining after Batch_01).

Class: `tooth` (id 0). **No FDI.** **No ICDAS.** YOLO boxes are **not** ground truth.

The previous Stage 3B round-robin 60-file list is archived in `stage3b_round_robin_archive/` (it was never annotated). This batch replaces it so human QC targets detector failure modes.

## Mouth views in this batch

- Frontal: 12
- Left_Lateral: 12
- Mandibular: 12
- Maxillary_Occlusal: 12
- Right_Lateral: 12

## Why these images

- 2 image(s) with **>24** YOLO detections (likely duplicates / false positives).
- 14 image(s) with **≤12** detections (possible missed teeth).
- Mix of low mean confidence, high low-conf fraction, and overlapping boxes.
- 10 pilot-style filenames and 50 distinct clinic patient IDs (plus unnamed pilots).
- Per-image scores: `selection_scores.csv`.

## How to annotate (CVAT)

1. Create task `iccc_batch_02` in project `iccc_whole_tooth_detection`.
2. Upload **only** the 60 originals in `seed_60/` (same pixels as `fdi_detection_dataset/images/selected/`; that folder was not modified).
3. Filename list: `cvat_upload_filenames.txt`.
4. Leave annotations **empty**. Draw whole-tooth rectangles from scratch (same as Batch_01).
5. Optionally keep `yolo_overlays_for_review/` open in a second window as a **checklist**, not as labels to copy.
6. Do **not** import `yolo_candidate_labels/` as ground truth.

Guidelines: `TOOTH_ANNOTATION_GUIDELINES.md`. Full Batch_01-style steps: `STAGE3C_MANUAL_ANNOTATION.md`.
