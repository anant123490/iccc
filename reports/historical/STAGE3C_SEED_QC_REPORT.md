# Stage 3C-1 — Seed annotation structural QC

Date: 2026-08-26

**STOP.** Human-verified Batch_01 boxes are **not in this workspace**. Training and pseudo-labeling were **not** started.

Machine-readable: `reports/stage3c_seed_qc.json`, `reports/stage3c_seed_qc.csv`.

Human annotations were **not** auto-edited. Empty Stage 3A/3B placeholders were **not** overwritten.

---

## What was inspected

| Item | Result |
| --- | --- |
| Batch_01 image list | **60** rows (`annotation_batches/Batch_01/image_list.csv`) |
| Images on disk (`images/selected/`) | **60 / 60** present |
| Annotation format | YOLO `.txt` placeholders + size-only Pascal VOC XML; COCO has no instances |
| YOLO files for Batch_01 | **60** exist, **60 empty** |
| VOC objects / `bndbox` | **0** |
| CVAT/Label Studio export (zip/json/nonempty YOLO) | **not found** in the project |
| `annotation_progress.csv` (Batch_01) | still `not_started` |
| Class IDs in nonempty labels | none (no boxes) |
| Non-`0` / FDI classes | **not present** (no labels to convert) |

---

## Box-level checks

Not applicable: **0** tooth boxes.

| Check | Result |
| --- | --- |
| Every Batch_01 image has a YOLO file | Yes (empty placeholders) |
| Image references exist | Yes |
| Positive width/height | n/a |
| Coordinates in bounds | n/a |
| NaN/Inf | n/a |
| Only class `0 = tooth` | no other class found; also **no class 0 boxes** |
| Duplicate identical boxes | n/a |

**structural_qc_pass = false**  
**stop_reason = NO_HUMAN_VERIFIED_BATCH_01_BOXES_IN_WORKSPACE**

Empty YOLO files are **not** treated as “annotated with zero teeth.” They are Stage 3A placeholders.

---

## What to put in the repo so this stage can continue

1. In CVAT, export task `iccc_batch_01` as **YOLO 1.1** (or COCO 1.0).
2. Copy the 60 nonempty `.txt` files into `fdi_detection_dataset/annotations/yolo/` using the **same stems** as the JPGs (do not replace images).
3. Confirm every line is `0 x_center y_center width height` (normalized). Do **not** export FDI class IDs.
4. Set Batch_01 rows in `annotation_project/annotation_progress.csv` to `done` and box counts.
5. Re-run Stage 3C-1.

Until then: **do not train**, **do not label the other 360 images**.
