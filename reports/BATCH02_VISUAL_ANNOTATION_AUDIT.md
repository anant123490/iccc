# Batch 02 visual annotation audit

**Date:** 2026-08-27  
**Mode:** Browser visualization only. **No training.** Labels and Batch 02 files were **not** rewritten.

## Viewer

| Item | Value |
|------|--------|
| App | `reports/tooth_detection_batch02_qc/browser_audit/app.py` |
| Catalog | `reports/tooth_detection_batch02_qc/browser_audit/selection.json` |
| Command (repo root, this machine) | `.venv\Scripts\streamlit.exe run reports\tooth_detection_batch02_qc\browser_audit\app.py --server.port 8501` |
| URL | **http://localhost:8501** |

Verified in browser: title “Batch 02 visual annotation audit”, catalog **100**, Previous/Next, Filter, metrics (split, tooth box count, class `tooth`, filename), green YOLO rectangles over the photo. Caption: “Existing YOLO rectangles (green)”. Boxes are read from the existing `.txt` files (e.g. image 23: `OHI_datasetIMG_6340_JPG.rf….txt`, 16 boxes, flag `border`).

If `streamlit` is not on PATH, use the `.venv\Scripts\streamlit.exe` form above.

## Number of images displayed

**100** (not all 1,063 KEEP images).

## Selection method

Fixed seed 42. Mixture written to `selection.json`:

| Reason | Count |
|--------|------:|
| many_teeth | 12 |
| few_teeth | 10 |
| border | 15 |
| extreme_aspect | 10 |
| tiny | 8 |
| low_fill (AABB aspect proxy; polygons not rewritten) | 8 |
| overlap | 12 |
| near_gray (from `held_out/review`) | 8 |
| gallery (from `held_out/review`) | 7 |
| random | 10 |

Splits in the 100: train 76, valid 13, test 11. Buckets: keep 85, review 15.

## Automated checks (on this 100, from existing rectangles)

- Every selected KEEP/review image had a matching `.txt`; boxes drawn as `0 xc yc w h` → pixel xyxy.
- Flags used for sampling: border-touching boxes, aspect > 5, area < 0.001, pairwise IoU ≥ 0.5, near-gray/gallery from cleanup CSV.
- These flags mean **possible** crowding, edge teeth, or elongated AABB — not automatic label errors.
- This audit does **not** certify the other 963 KEEP images.

## Visualization issues

- Streamlit top chrome can intercept clicks on Previous/Next after scrolling the image; scroll back up to navigate, or use the Filter dropdown.
- Dark theme: green boxes are high-contrast on teeth.
- First page load can show a skeleton before the overlay appears.

## Did the viewer render annotations?

**Yes.** Browser check showed an occlusal RGB photo with multiple green rectangles around individual teeth, sourced from the existing YOLO file.

## Recommendation

**PASS WITH MINOR ISSUES**

Rectangles on the inspected sample surround whole teeth and are suitable to **start** detector training on `batch02_clean`, with the known AABB limits (gum/neighbor overlap, border teeth, Roboflow stretch). Do not treat all 1,063 images as verified. Optional: click through the 100 in the local viewer before training.
