# Generated tooth crops

Crops are derived from the preserved full-mouth source photos by the whole-tooth detector. They are not ICDAS ground truth.

The previous generated crop set (5,676 JPGs), its 420 overlays, crop manifest, and prior ICDAS batch-prediction results were cleared on 2026-11-20. The 420 original RGB source photos, detector weights, detector annotations, and crop-generation code remain. Run python tools/run_tooth_cropping.py to regenerate crops when needed.

The detector_predictions/ folder retains separate tooth-detector confidence/label/visualization outputs. Those are not ICDAS labels. Do not treat any model prediction as clinical ground truth.

To label a crop for ICDAS:

1. A clinician assigns grade 0–4 in tools/label_icdas.py or tools/icdas_labeling_app.py.
2. Only then place it into the new dataset split after patient/source-level separation.

Active cropper: ml/src/tooth_cropping.py (output default: generated/).
