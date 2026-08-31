# Tooth detection dataset

This folder is for **full-mouth RGB intraoral photographs** and **rectangular whole-tooth bounding boxes**.

It is **not** the ICDAS dataset. Individual tooth crops belong in `data/tooth_crops/` (generated) or `data/icdas/` (clinician-labeled).

FDI tooth numbering is **out of scope**. Boxes are class `tooth` only.

## Canonical Batch 01 (do not flatten or overwrite)

Batch 01 already exists. It was **left in place** because training, cropping, and annotation JSON still point at this path:

| What | Location |
|------|----------|
| 420 original RGB photographs | `fdi_detection_dataset/images/selected/` |
| Batch 01 YOLO split (46 / 6 / 8) | `fdi_detection_dataset/tooth_detector_batch01/` |
| 767 verified tooth boxes | `fdi_detection_dataset/tooth_detector_batch01/labels/{train,val,test}/` |
| Current detector weights | `models/detection/tooth_detector_batch01/weights/best.pt` |
| Human QC batches | `annotation_batches/` |
| CVAT / Label Studio configs | `annotation_project/` |

The folder name `fdi_detection_dataset/` is **historical**. It is not an FDI labeling project.

## Future data entry (Batch 02+)

### 1. Where to put new RGB photographs

Put **new** full-mouth JPGs here:

```text
data/detection/raw_images/
  new_photo_001.jpg
  new_photo_002.jpg
```

Do **not** drop new files into `fdi_detection_dataset/images/selected/` unless you intend to extend the original 420-image set with a documented process. Prefer `raw_images/` so Batch 01 provenance stays intact.

### 2. Where to put bounding-box annotations

Put matching YOLO `.txt` files here (one file per image, same stem):

```text
data/detection/annotations/
  new_photo_001.txt
  new_photo_002.txt
```

YOLO format, one row per tooth, class `0` = tooth:

```text
0  <cx> <cy> <w> <h>
```

Values are normalized to `[0, 1]` (center x, center y, width, height).

### 3. How annotations correspond to images

- `new_photo_001.jpg` ↔ `new_photo_001.txt`
- Same filename stem, different folder
- Empty `.txt` means “reviewed, no tooth boxes” (rare). Missing `.txt` means “not annotated yet”

### 4. How to add images without destroying Batch 01

- Never replace files under `fdi_detection_dataset/tooth_detector_batch01/`
- Never overwrite `models/detection/tooth_detector_batch01/weights/best.pt`
- Keep new work under `data/detection/` and a **new** batch folder

### 5. How to create Batch 02

1. Copy or convert `raw_images/` + `annotations/` into a YOLO layout:

```text
data/detection/batches/batch02/
  images/train|val|test/
  labels/train|val|test/
  data.yaml
  metadata.json    # date, source, license, creator
```

2. `data.yaml` example:

```yaml
path: <absolute or repo-relative path to this batch02 folder>
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: tooth
```

3. Record split information and source/license in `metadata.json`.

### 6. How to create a new train/val/test split

- Split **by patient / source image**, not by individual boxes
- Do not put the same photograph in two splits
- Typical seed split for small sets: keep a held-out test set; do not mix Batch 01 test images into Batch 02 training unless you document it
- Write the split lists into the batch folder (CSV or YAML)

### 7. How to train a new YOLO model

```text
python tools/train_tooth_detector_new_batch.py --batch 02
```

This writes:

```text
models/detection/tooth_detector_batch02/
```

It **refuses** to train into `tooth_detector_batch01`.

Retrain Batch 01 only if you explicitly pass:

```text
python tools/train_tooth_detector_batch01.py --force-retrain-batch01
```

That command archives the current Batch 01 run under `archive/experiments/yolo_run_archives/` first.

### 8. How to evaluate

```text
python tools/verify_tooth_detector_batch01.py   # Batch 01 integrity
python tools/report_tooth_detector_train.py     # metrics report helper
```

For a new batch, run Ultralytics `val` on that batch’s `data.yaml` and store plots next to `models/detection/tooth_detector_batchNN/`.

### 9. How to preserve the old `best.pt`

Leave `models/detection/tooth_detector_batch01/weights/best.pt` untouched. New training must use a new `name=` directory. If you must retrain Batch 01, the train script copies the previous run into `archive/experiments/yolo_run_archives/` first.

## Empty folders in this directory

`raw_images/`, `annotations/`, `train/`, `val/`, and `test/` are **future intake** folders. They are empty on purpose so new data is not mixed into Batch 01.

## Related code

- Train Batch 01: `tools/train_tooth_detector_batch01.py`
- Crop teeth: `ml/src/tooth_cropping.py` (source images still read from `fdi_detection_dataset/images/selected/`)
- ML pointers: `ml/detection/`
