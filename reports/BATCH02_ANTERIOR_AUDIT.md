# Batch 02 targeted anterior / overlap annotation audit

**Date:** 2026-08-27  
**Mode:** Visual audit only. **No training.** Dataset labels were **not** rewritten. Audit buttons write only to `reports/tooth_detection_batch02_qc/anterior_audit/manual_reviews.json`.

## Dataset (read-only)

`data/detection/batches/batch02_clean/` — KEEP splits `train` / `valid` / `test` only.  
Original Batch 02 polygons, `yolo_detection/`, held-out review/exclude, Batch 01, ICDAS, and models were not modified.

## Viewer

| Item | Value |
|------|--------|
| App | `reports/tooth_detection_batch02_qc/anterior_audit/app.py` |
| Catalog | `reports/tooth_detection_batch02_qc/anterior_audit/selection.json` |
| Manual notes | `reports/tooth_detection_batch02_qc/anterior_audit/manual_reviews.json` (created on first mark) |
| Command | `.venv\Scripts\streamlit.exe run reports\tooth_detection_batch02_qc\anterior_audit\app.py --server.port 8502` |
| URL | **http://localhost:8502** |

Each page shows: original image, existing YOLO rectangles (green), filename, split, box count, Previous / Next, current index, sample reason, and geometry hints (max IoU, IoU≥0.5/0.7 pair counts, anterior-overlap pairs, border boxes, oversized vs median).

**GOOD / QUESTIONABLE / BAD** and the short note are stored only in `manual_reviews.json`. They never write `.txt` labels.

If Streamlit intercepts clicks after you scroll the photo, scroll back to the top for Previous / Next.

## Number of images in this targeted catalog

**100** (not a random sample; not all 1,063 KEEP images).

## Selection method (targeted)

All 1,063 KEEP images were scored from **existing** YOLO boxes. Images were taken in this order (skipping duplicates already picked):

| Sample reason | Count | What it prioritizes |
|---------------|------:|---------------------|
| `high_iou_duplicate` | 1 | Pairwise IoU ≥ 0.7 (near-duplicate stacked boxes). Only **one** KEEP image met this threshold. |
| `overlap_pairs` | 18 | Highest counts of IoU ≥ 0.5 neighbor pairs |
| `anterior_crowded` | 15 | Overlap (IoU ≥ 0.35) among boxes whose centers sit in the horizontal mid-band (proxy for incisor/canine crowding) |
| `tiny_packed` | 10 | Many very small boxes (area &lt; 0.001) |
| `many_detections` | 12 | Highest box counts |
| `extreme_aspect` | 10 | Aspect ratio &gt; 5 |
| `oversized_vs_neighbors` | 10 | Boxes &gt; 2.5× median box area on the same image |
| `border_touching` | 13 | Boxes flush with the image edge |
| `max_iou_fill` | 11 | Remaining slots: highest max pairwise IoU |

This is **not** random. IoU ≥ 0.7 stacked duplicates are rare on KEEP; most “stacked anterior” risk shows up as IoU 0.35–0.5.

## Automated geometry screen (this 100 only)

These are **not** human GOOD/BAD marks. They flag possible issues for the viewer.

| Flag | Count of images | Definition used |
|------|----------------:|-----------------|
| Duplicate-looking boxes | **1** | ≥1 pair with IoU ≥ 0.7 |
| Multi-tooth box candidate | **30** | ≥1 box with area &gt; 2.5× the image’s median box area |
| Excessive non-tooth content (proxy) | **4** | Both oversized-vs-median **and** extreme aspect on the same image |
| Border-truncated teeth | **16** | ≥1 box touching the image border |

IoU ≥ 0.5 overlap pairs are common in the overlap/anterior buckets by construction; they are crowding, not proven duplicates.

## Human marks (GOOD / QUESTIONABLE / BAD)

Marks exist only after someone clicks the buttons in the viewer.

| Mark | Count |
|------|------:|
| Images in catalog | 100 |
| **GOOD** | **0** |
| **QUESTIONABLE** | **0** |
| **BAD** | **0** |
| Unrated | **100** |

Re-open the report after marking; counts live in `manual_reviews.json`.

## Visualization check

The viewer draws `0 xc yc w h` from the matching KEEP `.txt` as pixel rectangles. It does not convert polygons and does not run inference.

## Recommendation

Use this catalog to inspect overlapping anterior boxes, neighbor coverage, oversized molars, and border-truncated teeth. Do not train until this targeted pass (and any follow-up edits you choose) is finished. **No training was run for this audit.**
