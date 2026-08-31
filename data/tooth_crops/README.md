# Generated tooth crops

Crops produced by the **whole-tooth detector**. They are **not** ICDAS ground truth.

| Folder | Meaning |
|--------|---------|
| `generated/` | Detector output (images, overlays, manifest). ~5,676 crops from 420 photos |
| `reviewed/` | Human-accepted crops **without** an ICDAS grade (optional QC) |
| `detector_predictions/` | YOLO txt / reports on remaining images; not Batch 01 GT |

Do **not** copy `generated/` into `data/icdas/train/` automatically.

Do **not** treat `detector_predictions/icdas_predictions/` as clinical labels (those scores came from the stale ordinal keras).

To label a crop for ICDAS:

1. Clinician assigns 0–4 in `tools/label_icdas.py` or `tools/icdas_labeling_app.py`
2. Only then copy into `data/icdas/train|val|test/<grade>/`

Active cropper: `ml/src/tooth_cropping.py` (output default: this `generated/` folder).
