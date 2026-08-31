# Batch 02 tooth-detection QC report

**Date:** 2026-08-27  
**Mode:** Preparation + QC only. **No training.** Batch 01, ICDAS, and models were not modified. Original Roboflow polygon `.txt` files were not overwritten.

**Recommendation:** `READY AFTER CLEANUP`

---

## Dataset

| Item | Count |
|------|------:|
| Total images | 2907 |
| train / valid / test | 2541 / 246 / 120 |
| Total tooth objects (converted) | **48942** (matches prior polygon audit) |
| Class | `0 = tooth` (`nc: 1`) |

Prepared copy:

`data/detection/batches/batch02/yolo_detection/`

Original polygons:

`C:\Users\anant\AppData\Local\Temp\icdas_inspect_intraoral_tooth_detection_v1i`  
ZIP: `C:\Users\anant\Downloads\Intraoral Tooth Detection.v1i.yolov8.zip` (unchanged)

Script: `tools/convert_tooth_polygons_to_yolo_boxes.py`

---

## Annotation

| Item | Detail |
|------|--------|
| Original format | YOLOv8 **polygons** (typically 5 vertices / 11 fields; some 6 vertices) |
| Converted format | YOLO detection: `0 x_center y_center width height` (normalized) |
| Method | Tight AABB: min/max polygon x,y → center, width, height; clamp to [0,1] |
| Conversion errors | **0** |
| Invalid 5-value lines | **0** |
| Image↔label mismatches | **0** |

Per split (converted boxes):

| Split | Images | Label files | Boxes | Expected |
|-------|--------|-------------|------:|---------:|
| train | 2541 | 2541 | 42779 | 42779 |
| valid | 246 | 246 | 4153 | 4153 |
| test | 120 | 120 | 2010 | 2010 |
| total | 2907 | 2907 | 48942 | 48942 |

Spot check: a source polygon file still has 11 fields; the converted twin has 5.

---

## Box quality

Normalized box stats (all 48,942):

| Metric | min | max | mean | median |
|--------|-----|-----|------|--------|
| Area (w×h) | 0.00035 | 0.117 | 0.0187 | 0.0158 |
| Width | 0.011 | 0.319 | 0.103 | 0.098 |
| Height | 0.019 | 0.495 | 0.168 | 0.163 |
| Polygon / box area | 0.276 | 1.00 | 0.869 | 0.922 |
| Aspect (max/min side) | 1.00 | 11.89 | 1.80 | 1.63 |

Flags (review list only; **nothing deleted**): `reports/tooth_detection_batch02_qc/suspicious_boxes.csv` — **861** rows covering **474** images.

| Flag | Count |
|------|------:|
| Touches image boundary | 691 |
| Extreme aspect (&gt;5) | 117 |
| Extremely small area (&lt;0.001) | 31 |
| Low polygon-to-box ratio (&lt;0.45) | 27 |
| Heavy overlap IoU ≥ 0.7 | 2 pairs (2 images) |
| Extremely large area (&gt;0.12) | 0 |
| Near-duplicate boxes IoU ≥ 0.92 | 0 |

Mean polygon-to-box ratio **0.87** (median **0.92**) means most rectangles are reasonably tight; the low-ratio tail is rotated/oblique teeth where AABB includes extra gum/neighbor.

Visual overlap of adjacent teeth is common but pairwise IoU ≥ 0.7 is rare (2 pairs). Max IoU observed ~0.71.

---

## Duplicate analysis

Roboflow applied crop/rotate/blur and “3 versions of each source.”

| Item | Count |
|------|------:|
| Total files | 2907 |
| Unique stems before `.rf.<hash>` | **1213** |
| Stems with &gt;1 file | 847 |
| Extra augmented copies | **1694** |
| Stems in more than one of train/valid/test | **0** |

**No train/validation/test leakage** by Roboflow stem. Duplicates stay **inside** a split (mostly train). Counts are inflated vs unique mouths/photos.

---

## Visual QC

Overlays (converted **rectangles** only): `reports/tooth_detection_batch02_qc/overlays/`

| Split | Samples |
|-------|--------:|
| train | 10 |
| valid | 5 |
| test | 5 |

Findings:

- Rectangles generally wrap **individual whole teeth**, not caries spots or retractors as a class.
- Axis-aligned boxes **do** include some gum, a sliver of the neighbor, and (on occlusal/rotated views) tongue or dark background in the corners.
- Some views label **one arch only** (source annotation, not a conversion bug).
- Retractors, lips, and face usually sit **outside** the boxes.
- Crowded anterior teeth show overlapping rectangles (expected for AABB on an arch).
- Roboflow **rotation + 640 stretch** produces black corners; some boxes sit near those borders (matches 691 boundary flags).

---

## Comparison with Batch 01 (`fdi_detection_dataset/images/selected/`)

Batch 01 was **not** modified.

| | Batch 01 selected (420) | Batch 02 Roboflow (2907) |
|--|-------------------------|---------------------------|
| Style | Color clinical RGB (0 near-gray) | 2624 color, **283 near-gray** |
| Gallery / 3D-like names | 0 | **149** `gallery`/`penta_` |
| Screenshots | 0 | 21 |
| Resolution | Native camera sizes | All **640×640 stretched** |
| Retractors / views | Clinical intraoral set | Mix of retractors, occlusal, frontal; some 3D-scan look |
| GT | 767 **human** boxes on **60** images | 48942 converted boxes on all 2907 files (incl. augs) |
| Domain vs our camera | Same pool as current detector | **Shifted** (resize, augs, gallery/gray subset) |

Useful as extra **tooth** boxes, not a drop-in clone of the 420-camera domain.

---

## Final recommendation

**`READY AFTER CLEANUP`**

Do **not** train yet. Before detector training:

1. Deduplicate to unique `.rf.` stems (keep one file per 1213 sources) **or** train with a documented “aug-in-train-only” policy.
2. Optionally drop near-gray, `gallery`/`penta_`, and screenshot files if the target is the RGB camera pipeline.
3. Human-spot the 861 flagged rows (boundary / tiny / extreme aspect / low fill).
4. Keep Batch 02 **separate** from Batch 01; do not mix splits blindly.

JSON: `reports/tooth_detection_batch02_qc/conversion_qc.json`
