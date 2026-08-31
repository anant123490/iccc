# Tooth crop origin audit

**Project:** CCC AI Dentist Camera 2.0  
**Date:** 2026-08-27  
**Mode:** AUDIT ONLY (no training, no data/model/code changes except this report)

---

## Classification

**`B — EXISTING DATA AVAILABLE BUT NEEDS PREPARATION`**

**Recommendation:** `BATCH 02 DECISION REQUIRES FURTHER QC`

The 5,676 crops are fully traceable. The original 420 full-mouth RGB photographs still exist. The boxes that produced the crops also still exist (in the crop manifest). Those boxes are **YOLO detector predictions**, not the Batch 01 human-verified 767 boxes. They are not ready to treat as a new ground-truth detection dataset without QC.

---

## 1. Crop-generation script

| Item | Value |
|------|--------|
| CLI | `tools/run_tooth_cropping.py` |
| Library | `ml/src/tooth_cropping.py` (`ToothCropPipeline`) |
| **Not** used for these 5,676 files | `tools/crop_teeth.py` (legacy cropper from existing YOLO/COCO/VOC boxes; different pipeline) |

`run_tooth_cropping.py` loads Batch 01 `best.pt`, runs Ultralytics `predict` on every JPG in the source folder, pads boxes 8%, quality-checks crops, and writes images + overlay + CSV.

Method: **axis-aligned bounding boxes from a detector** (class `tooth`, confidence ≥ 0.25). Not segmentation masks. Not human GT boxes. Not FDI. Not ICDAS.

Associated files:

- `data/tooth_crops/generated/run_summary.json`
- `data/tooth_crops/generated/manifest.csv`
- `data/tooth_crops/generated/skipped.csv` (header only; 0 skipped rows)
- `reports/TOOTH_CROPPING_REPORT.md`

There is **no** `crops.csv` in `generated/` (that filename belongs to `crop_teeth.py`).

Weights path recorded at generation time:

`models/tooth_detector_batch01/weights/best.pt`

(Those weights now live at `models/detection/tooth_detector_batch01/weights/best.pt` after repository organization. The file was not re-read or overwritten for this audit.)

---

## 2. Original source dataset / path

From `run_summary.json`:

```text
source: .../fdi_detection_dataset/images/selected
images_processed: 420
```

Folder name `fdi_detection_dataset` is historical. Contents used here are **RGB intraoral photographs**, not FDI labels.

---

## 3. Number of crops

| Item | Count |
|------|------:|
| Manifest rows | 5676 |
| JPG files in `generated/images/` | 5676 |
| Names in manifest missing on disk | 0 |
| Files on disk missing from manifest | 0 |
| Unreadable crop JPGs | 0 |
| Overlay JPGs | 420 |

Matches `run_summary.json` `total_tooth_crops: 5676`. Raw detections before QC also 5676 (nothing skipped).

Manifest columns:

`image_name, crop_name, confidence, x1, y1, x2, y2, crop_w, crop_h`

- `image_name` = full-mouth JPG basename  
- `x1,y1,x2,y2` = **padded** pixel rectangles used to cut the crop  
- `confidence` = detector score (0.2502–0.9103, mean ~0.556)  
- No split column, no annotation ID, no ICDAS class, no FDI  

---

## 4. Unique source images

| Item | Count |
|------|------:|
| Unique `image_name` in manifest | **420** |
| Crops per source | min **6**, max **27**, mean **13.514** |
| Sources with a matching file in `fdi_detection_dataset/images/selected/` | **420 / 420** |
| Sources whose original cannot be found | **0** |

Every crop maps to an original full-mouth image by basename.

---

## 5. Original full-mouth RGB images found

| Location | Result |
|----------|--------|
| `fdi_detection_dataset/images/selected/` | **420** `.jpg`, all readable |
| `data/detection/raw_images/` | Future intake; **not** the crop source |
| `annotation_batches/Batch_01/seed_60/` | **60** copies of Batch 01 images (subset of the 420) |

All 420 selected JPGs verified openable. Formats: JPEG only.

These 420 are the same set the detector was run on. They correspond 1:1 to the 5,676 crops.

---

## 6–10. Original annotations / format / boxes / classes / whole teeth

### What actually created the 5,676 crops

**Detector bounding boxes**, stored as pixel xyxy in `manifest.csv`.

- Format: axis-aligned rectangles (after 8% pad), class implied `tooth`  
- Class names: **`tooth` only** (pipeline ignores any class id ≠ 0)  
- Total boxes that became crops: **5,676**  
- They represent **candidate whole-tooth regions**, not lesions, gums, or instruments by design  
- They are **not clinician-verified** (2,244 of 5,676 have confidence &lt; 0.5)

Those boxes **can** recut the same crops from the 420 originals (coordinates are in the manifest). That does **not** make them training GT.

### Human-verified tooth boxes (Batch 01) — different, smaller set

| Item | Value |
|------|--------|
| Path | `fdi_detection_dataset/tooth_detector_batch01/` |
| Format | YOLO txt `class_id cx cy w h` (normalized) |
| Images | 46 train / 6 val / 8 test = **60** |
| Boxes | **767** |
| Class | `0: tooth` (`data.yaml` `nc: 1`) |

All 60 Batch 01 images are among the 420 crop sources. The other **360** crop sources have **no** Batch 01 human labels.

Placeholder YOLO files under `fdi_detection_dataset/annotations/yolo/` are not this crop run’s source.

`tools/crop_teeth.py` did not generate these 5,676 files.

---

## 11. Whether crops map back to original images

**Yes.** 5,676 / 5,676 manifest rows resolve to a file in `fdi_detection_dataset/images/selected/`. Crop filenames are `{source_stem}_tooth_{index:03d}.jpg`.

---

## 12. Whether existing data can train a YOLO tooth detector

| Data | Usable as detector GT? |
|------|-------------------------|
| 420 full-mouth RGB JPGs | Yes, as **images** |
| 767 Batch 01 boxes on 60 images | Yes — **already used**; this **is** Batch 01 |
| 5,676 crop boxes in the manifest | **No, not as reliable GT** without human QC (self-training on Batch 01 detector outputs) |

You can train a detector **today** only on the existing Batch 01 split. Using the 5,676 predicted boxes as labels would recycle the same model’s errors.

---

## 13. Relationship to Batch 01

The 5,676 crops **came from running the Batch 01 detector on all 420 selected photos**, not from cropping the 767 GT boxes.

```text
Batch 01 human boxes (60 images, 767 boxes)
    → train YOLO11n
    → models/.../tooth_detector_batch01/weights/best.pt
    → infer on 420 selected JPGs
    → 5,676 crops + overlays + manifest
```

So:

- Same **image pool** as Batch 01’s 420 originals  
- **Not** the same annotation set as Batch 01  
- Crops are **generated detections**, documented as not ICDAS GT  

---

## 14. Newly downloaded dataset vs this source

Inspected earlier (not re-extracted in this audit):

`Intraoral Tooth Detection.v1i.yolov8.zip` (Roboflow, class `tooth`, polygons, 2,907 images).

| Question | Answer |
|----------|--------|
| Needed to explain the 5,676 crops? | **No** |
| Same images as the 420 selected photos? | **No** (different Roboflow/OHI export) |
| Replaces recovering originals for these crops? | **No** — originals are already in-repo |
| Needed for Batch 02? | **Only if** you want extra **external** tooth boxes after conversion/QC. It does not fill missing GT on the 360 unlabeled photos of **this** 420-set |

---

## Conclusion

**`B — EXISTING DATA AVAILABLE BUT NEEDS PREPARATION`**

- Full-mouth RGB: **present** (420 JPG).  
- Boxes that made the crops: **present** (manifest xyxy) but **detector-predicted**.  
- Reliable whole-tooth GT: **only Batch 01** (60 images / 767 boxes).  

**`BATCH 02 DECISION REQUIRES FURTHER QC`**

- Do **not** treat the 5,676 predicted boxes as Batch 02 GT.  
- Do **not** need the Roboflow ZIP to recover this crop lineage.  
- A new Batch 02 is **necessary** only if the goal is more **human** whole-tooth boxes than the current 60-image set (QC remaining 360 of the 420, and/or convert+review an external set separately).  
- If the goal is only to keep using the current detector, Batch 02 is **not** required to explain or keep the 5,676 crops.
