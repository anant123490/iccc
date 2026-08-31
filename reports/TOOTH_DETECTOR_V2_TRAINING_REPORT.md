# Tooth Detector V2 training report

**Mode:** Training a **new** YOLO11n detector. Batch 01 weights, Batch 02 CLEAN, GOOD folders, ICDAS, frontend, and backend were **not** modified.

**Best model:** `models/detection/tooth_detector_v2/weights/best.pt`  
**Last checkpoint:** `models/detection/tooth_detector_v2/weights/last.pt`

## Dataset

| Item | Value |
|------|--------|
| Path | `data/detection/gold_detector_dataset/` |
| Unique images | **176** |
| Tooth boxes | **3633** |
| Train | 124 images / 2544 boxes |
| Valid | 26 images / 577 boxes |
| Test | 26 images / 512 boxes |
| Class | `0 = tooth` (`nc: 1`) |

Gold validation before this run: 0 label errors, 0 split leakage, 0 duplicates. Test was **not** used for training or early stopping.

## Environment

| Item | Value |
|------|--------|
| Python | 3.12.5 (system: `Python312\python.exe`) — same stack as Batch 01 |
| Ultralytics | 8.4.45 |
| Torch | 2.2.2+cpu |
| Device | **CPU** (12th Gen Intel Core i5-12450H) |
| Init weights | `yolo11n.pt` (downloaded to repo root for this run) |

`.venv` does not include `torch`/`ultralytics`; training reused the Batch 01 Python, not a version bump.

## Training

| Item | Value |
|------|--------|
| Architecture | YOLO11n, `single_cls=True` |
| Data | `data/detection/gold_detector_dataset/data.yaml` |
| Image size | 640 |
| Batch | 8 |
| Epochs | 100 (patience 20; ran **all 100**) |
| Optimizer | AdamW, `lr0=0.001` |
| Seed | 42 |
| Wall time | **5.283 hours** |
| Save dir | `models/detection/tooth_detector_v2/` |

Best logged val mAP50-95 during training: **epoch 76** (0.586). Final `best.pt` eval is below.

## Validation (`best.pt` on Gold valid)

| Metric | V2 |
|--------|-----|
| Precision | **0.9448** |
| Recall | **0.8894** |
| F1 | **0.9162** |
| mAP50 | **0.9576** |
| mAP50-95 | **0.5875** |

## Test (`best.pt` on Gold test — held out)

| Metric | V2 |
|--------|-----|
| Precision | **0.8763** |
| Recall | **0.8856** |
| F1 | **0.8809** |
| mAP50 | **0.9183** |
| mAP50-95 | **0.5511** |

## Comparison with Batch 01 (not replaced)

Batch 01 original numbers were measured on the **tiny Batch 01 val/test** (6 / 8 images). A fair comparison for crop-pipeline use is **both models on the Gold splits**.

| Split | Model | Precision | Recall | F1 | mAP50 | mAP50-95 |
|-------|--------|-----------|--------|-----|-------|----------|
| Gold val | **V2** | 0.945 | 0.889 | 0.916 | 0.958 | **0.587** |
| Gold val | Batch 01 | 0.459 | 0.350 | 0.397 | 0.305 | 0.066 |
| Gold test | **V2** | 0.876 | 0.886 | 0.881 | 0.918 | **0.551** |
| Gold test | Batch 01 | 0.449 | 0.396 | 0.421 | 0.325 | 0.080 |
| B01 val (legacy) | Batch 01 | 0.754 | 0.736 | 0.745 | 0.745 | 0.283 |
| B01 test (legacy) | Batch 01 | 0.700 | 0.726 | 0.712 | 0.718 | 0.282 |

### Improvements

- Much higher mAP and F1 on Gold (Batch 02-style 640×640 + Batch 01 GT mixed). Batch 01 was trained on 46 native camera photos only; it **under-detects** on the Gold mix.
- Recall on Gold val ~0.89 vs ~0.35 for Batch 01 — fewer missed teeth on this distribution.
- Localization (mAP50-95) roughly **2×** Batch 01’s legacy B01-test figure and far above Batch 01 on Gold.

### Regressions / remaining issues

- Overlay class text shows **`item`** (Ultralytics `single_cls` display). The trained class is still the single **tooth** head; crops should treat every box as tooth.
- Precision on Gold test (0.88) is below Gold val (0.95): more false positives or duplicate boxes on the test mix.
- Crowded anterior views still show **overlapping AABBs** and occasional extra boxes (same as Gold GT style). That can yield duplicate crops unless NMS/`iou` is tuned at inference.
- Border / posterior teeth: lower confidence (often 0.3–0.6); some dark corner molars still missed.
- Boxes still include some gum (expected for whole-tooth AABB crops).

Batch 01 remains the better **documented** detector **only on its original 8-image camera test**. V2 is the better detector **on the Gold dataset** and is the candidate for the next real-world camera test. **Do not overwrite Batch 01.**

## Visual QC

Overlays (predictions + confidence, source images unchanged):

- 10 validation: `reports/tooth_detector_v2_visual_test/val_overlays/`
- 10 test: `reports/tooth_detector_v2_visual_test/test_overlays/`

Observed: high-confidence frontal/retractor views with one box per visible crown; overlap on incisors; a few low-confidence edge boxes; some posterior misses on Batch 01-style frontal photos with fingers in frame.

## Real-world test (next)

Folder: `reports/tooth_detector_v2_real_world_test/`  
Instructions: `reports/tooth_detector_v2_real_world_test/README.md`  
Put **20–30 new** clinic photos in `incoming/` (not Gold images). Inference command is in that README.

## Metrics JSON

`models/detection/tooth_detector_v2/eval_metrics.json`
