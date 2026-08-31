# Stage 3C — Whole-tooth bounding-box annotation

Date: 2026-08-26

**Result: machine-generated candidate boxes were NOT created.**

This is required by Stage 3C Step 2: there is **no reliable local whole-tooth detector** in the repository or environment that can be used without fabricating labels, converting lesions, or downloading an unrelated model.

Empty YOLO / COCO / VOC placeholders from Stage 3A/3B were **left empty**. They are **not** ground truth.

---

## Stage 3B verification

| Check | Result |
| --- | --- |
| Selected images | **420** (expected 420; **no mismatch**) |
| Readable | **420** / unreadable **0** |
| Class config | **exactly** `0 = tooth` (`classes.yaml`) |
| Pre-existing nonempty YOLO tooth labels | **0** (nothing overwritten) |
| Guidelines present | `TOOTH_ANNOTATION_GUIDELINES.md`, `ANNOTATION_QC_CHECKLIST.md`, `README_STAGE3B.md` |
| ICDAS `dataset/train` | still absent; `dataset/annotations.csv` not written by this stage |

---

## Local method inspection

| Resource | Finding |
| --- | --- |
| Tooth / YOLO `.pt` / ONNX in repo | **None** |
| `models/*.keras` | ICDAS **classification** only — cannot output tooth boxes |
| `ultralytics` / `torch` installed | Yes, but **no tooth checkpoint**. Using COCO YOLO would invent boxes and typically **download** weights — forbidden |
| OpenCV | Installed; contour hacks would **fabricate** non-tooth boxes — not used |
| Lesion Pascal VOC `d`/`D` | **Not used** (lesions ≠ teeth) |
| Internet / extra datasets | **Not used** |

**Decision:** do not invent bounding boxes. **Manual annotation required.**

---

## Candidate vs verified

| Kind | Count |
| --- | --- |
| Machine **candidate** tooth boxes | **0** |
| Human **verified** tooth boxes | **0** |

Do not claim automatic labels as ground truth.

---

## Annotation files (unchanged placeholders)

| Format | Status |
| --- | --- |
| YOLO | 420 empty `.txt` — valid zero-object files; **not** annotated teeth |
| COCO | `instances_placeholder.json` — `annotations: []` |
| Pascal VOC | size-only XML; **no** `<object>` tooth boxes |

Visualizations were **not** written (no boxes; originals not duplicated). See `fdi_detection_dataset/review/annotated_visualizations/README.md`.

---

## QC

Automated checks on the **empty** label set:

- Every selected image has a YOLO file: **yes**
- Class IDs other than 0: **none** (no lines)
- Invalid / NaN / duplicate boxes: **0**
- Empty annotations: **420** (expected given the stop)
- Images flagged for review / manual annotation: **420**

CSVs:

- `reports/stage3c_annotation_qc.csv`
- `reports/stage3c_manual_review.csv`

---

## Statistics

| Metric | Result |
| --- | --- |
| Total images | 420 |
| Candidate tooth boxes | 0 |
| Mean / median / min / max boxes per image | 0 / 0 / 0 / 0 |
| Images with zero boxes | 420 |
| High box-count images | 0 |
| Confidence | n/a (no detector) |
| Review count | 420 |

**By batch (candidate boxes):** Batch_01 … Batch_07 = **0** each.

---

## Protections

| Item | Result |
| --- | --- |
| Lesion XML converted | **NO** |
| FDI labels (11–48 as classes) | **0** |
| Detection classes | **1** (`tooth`) |
| ICDAS modified | **NO** |
| YOLO trained | **NO** |
| TensorFlow trained | **NO** |
| Original 6,265 images modified | **NO** |
| `dataset/` / `ml/` / `models/` / backend / frontend | **not modified by this stage** |

---

## Final summary table

| Metric | Result |
| --- | ---: |
| Selected images | 420 |
| Successfully processed (auto boxes) | 0 |
| Candidate tooth boxes | 0 |
| Images with zero boxes | 420 |
| Images requiring review | 420 |
| Invalid annotations | 0 |
| Duplicate boxes flagged | 0 |
| YOLO annotations | 420 empty files |
| COCO annotations | 0 objects |
| Pascal VOC annotations | 0 tooth objects |
| Detection classes | 1 |
| FDI labels | 0 |
| ICDAS modified | NO |
| Models trained | NO |

---

## Final questions

1. Were all 420 selected images processed? **Inventoried and QC-flagged: yes. Auto-boxed: no.**  
2. How many candidate tooth boxes were generated? **0**  
3. How many images require human review? **420** (manual annotation, not machine-box QC)  
4. Are the YOLO annotations structurally valid? **Yes as empty files; they do not contain teeth.**  
5. Is the COCO file valid? **Yes as a placeholder with empty `annotations`.**  
6. Were any lesion annotations converted? **NO**  
7. Were any FDI labels generated? **NO**  
8. Was ICDAS modified? **NO**  
9. Was any model trained? **NO**  
10. Are the annotations ready for human QC? **Ready for human *annotation* via Stage 3B CVAT/Label Studio. Not ready as a machine-candidate QC set, because no candidates exist.**

---

## Next step

Draw `tooth` boxes in CVAT or Label Studio using `README_STAGE3B.md` and `TOOTH_ANNOTATION_GUIDELINES.md`, then **Stage 3D — Annotation Quality Control and Dataset Finalization**.

**STOP.** No FDI, no ICDAS, no training, no lesion conversion, no fabricated boxes.
