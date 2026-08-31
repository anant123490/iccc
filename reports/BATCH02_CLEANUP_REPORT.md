# Batch 02 cleanup report

**Date:** 2026-08-27  
**Mode:** Cleanup copy only. **No training.** Batch 01, ICDAS, models, original polygons, and `data/detection/batches/batch02/yolo_detection/` were not modified.

**Recommendation:** `READY FOR TRAINING`

(Train later on `batch02_clean` only. Do not train in this step.)

---

## What changed

| Location | Role |
|----------|------|
| `data/detection/batches/batch02/yolo_detection/` | **Unchanged** converted Batch 02 |
| `data/detection/batches/batch02_clean/` | New KEEP set (unique stems, color clinical-like) |
| `batch02_clean/held_out/review/` | REVIEW copies (not in `data.yaml` splits) |
| `batch02_clean/held_out/excluded/` | EXCLUDE copies (augmentation duplicates) |
| `tools/cleanup_batch02.py` | Reproducible cleanup |

Classification table: `data/detection/batches/batch02_clean/file_classification.csv`

---

## Unique stems vs Roboflow copies

| Item | Count |
|------|------:|
| Original converted images | 2907 |
| Unique source stems (before `.rf.<hash>`) | **1213** |
| Extra Roboflow copies | **1694** |
| Stems spanning train/valid/test | 0 (unchanged) |

For each stem, **one** representative was kept: the file whose tooth-box count is closest to the group median (tie: filename). Other copies → **EXCLUDE** (`roboflow_augmentation_duplicate`).

---

## KEEP / REVIEW / EXCLUDE

Unusual images were **not** deleted just for being unusual. They were classified:

| Class | Rule | Count |
|-------|------|------:|
| KEEP | Unique representative, color, not gallery/screenshot | **1063** |
| REVIEW | Unique representative that is near-gray, `gallery`/`penta_`, or `screen_shot` | **150** |
| EXCLUDE | Non-chosen Roboflow copies | **1694** |

### Near-gray (283 files in original Batch 02)

- Not auto-deleted.
- Unique representatives → **REVIEW** (`near_grayscale_color_stats`): **88** images in `held_out/review/`.
- Remaining near-gray files were extra augs → EXCLUDE with the duplicate rule.
- **0** near-gray images in the KEEP training splits.

### Gallery / 3D-like (149 files)

- Not auto-deleted.
- Unique representatives → **REVIEW** (`gallery_or_3d_like_filename`): **63** in `held_out/review/`.
- Extra copies → EXCLUDE.

### Screenshots (21 files)

- Unique representatives → **REVIEW** (`screenshot_filename`).
- Overlap with gray/gallery is possible; one class per file (screenshot/gallery checked first).

### Border-touching boxes

- Images were **not** dropped for a tooth touching the frame.
- Extra flags only for clearly huge multi-border boxes (`w≥0.28` and `h≥0.40`, or area≥0.08 on ≥2 borders).
- **0** KEEP images matched that bar. Routine edge teeth stay in KEEP.

---

## Clean training set (`data.yaml`)

Layout:

```text
batch02_clean/
  images/train|valid|test/
  labels/train|valid|test/
  data.yaml
```

| Split | Images | Label files | Tooth boxes | Missing labels |
|-------|--------|-------------|-------------|----------------|
| train | 720 | 720 | 12112 | 0 |
| valid | 228 | 228 | 3865 | 0 |
| test | 115 | 115 | 1912 | 0 |
| **total** | **1063** | **1063** | **17889** | **0** |

| Item | Value |
|------|--------|
| Class counts | `0 = tooth` only, **17889** boxes |
| Duplicate count in clean splits | **0** extra copies (1063 unique stems) |
| Excluded count | **1694** |
| Review count | **150** |
| Every KEEP image has a matching YOLO rectangle file | **Yes** |

---

## Final recommendation

**`READY FOR TRAINING`**

Use `data/detection/batches/batch02_clean/data.yaml` when you later train a **new** detector (e.g. `tooth_detector_batch02`). Do **not** overwrite Batch 01 weights.

Still optional: human pass on the **150** REVIEW images before mixing them back in. They are already copied under `held_out/review/` and are **not** in the training splits.
