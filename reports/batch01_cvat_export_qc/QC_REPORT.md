# Batch_01 CVAT Ultralytics YOLO export — QC

Date: 2026-08-27  
ZIP inspected only (not copied into `fdi_detection_dataset/annotations/yolo/`).  
Originals in `fdi_detection_dataset/images/selected/` were **read** for overlay copies; they were **not** modified. No training. ICDAS / backend / checkpoints untouched.

ZIP: `C:\Users\anant\Downloads\task_2546747_annotations_2026_08_26_20_27_14_ultralytics yolo detection 1.0.zip` (46,982 bytes)

---

## Checks

| Check | Result |
| --- | --- |
| Images **inside** the ZIP | **0** (labels-only export: `data.yaml` + `train.txt` + 60 label files) |
| Image **names** in `train.txt` | **60** |
| Matching JPGs on disk (`images/selected/`) | **60 / 60** (read-only) |
| Annotation `.txt` files | **60** |
| Stems vs Batch_01 `cvat_upload_filenames.txt` | **exact match** |
| `data.yaml` classes | `0: tooth` only |
| Class IDs in labels | **only 0** (767 lines) |
| Empty label files | **0** |
| Malformed lines (need 5 fields) | **0** |
| Non-finite / non-positive w,h | **0** |
| Normalized coords outside [0,1] | **0** |
| Pixel boxes outside image | **0** |
| Duplicate identical boxes | **0** |
| Near-full-image boxes | **0** |
| Tiny boxes (area &lt; 20×20 px) | **1** (flag only; coords still valid) |

**Annotation verification: PASS**  
**“60 image files packed in the ZIP”: FAIL** — CVAT Ultralytics Detection 1.0 often ships **labels only**. The 60 photos are already in `selected/`; `train.txt` points at `data/images/train/<name>.jpg` which is **not** inside this archive.

Do **not** treat this as a reason the boxes are wrong. Do **not** train until you decide to import the 60 `.txt` files next to the existing selected JPGs (separate step).

---

## Tooth count

**767** whole-tooth boxes (class `tooth` / 0).  
Per image: **min 7, max 22**. Mean ≈ 12.8.

No FDI classes. No ICDAS. No `d`/`D`.

---

## Sample visualizations

Overlays (green `tooth` boxes) saved as **new** files only:

`reports/batch01_cvat_export_qc/visualizations/`

10 samples (2 per Batch_01 view). Original JPGs unchanged.

Per-image table: `reports/batch01_cvat_export_qc/per_image.csv`  
JSON: `reports/batch01_cvat_export_qc/qc_summary.json`

---

## Training

**Not started.** Import of labels into the detection dataset is a later explicit step.
