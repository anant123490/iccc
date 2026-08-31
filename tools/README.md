# Operator tools

Detection, cropping, ICDAS labeling, and audits. Ingest scripts: `tools/ingest/`.

Lesion d/D trainers were moved to `archive/experiments/caries_lesion/` (FastAPI caries modules stay in `app/backend/`).

## Tooth detection

```text
python tools/verify_tooth_detector_batch01.py
python tools/train_tooth_detector_new_batch.py --batch 02
python tools/run_tooth_cropping.py
```

Batch 01 retrain is blocked unless you pass `--force-retrain-batch01`.

## ICDAS labels (human only)

```text
streamlit run tools/label_icdas.py
python tools/build_dataset.py
python tools/check_dataset.py
python ml/train.py --config ml/configs/default.yaml
```

Do not train until `data/icdas/train|val|test` contain clinician-confirmed pixels.

Crops used for labeling live in `data/tooth_crops/generated/`. They are not ICDAS GT until labeled.

## Legacy box cropper

`tools/crop_teeth.py` crops from existing boxes (YOLO/COCO/VOC). It does not assign ICDAS. Prefer `ml/src/tooth_cropping.py` for the camera detector.
