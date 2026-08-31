# Tooth Detector V2 — real-world inference (unseen photos)

This folder is for **new intraoral photos that were not used in Gold train/valid/test**.

Do **not** copy Gold dataset images here. Do **not** use Batch 01 training photos.

## What to put here

1. Copy **20–30 new** RGB intraoral photos into `incoming/` (JPG/PNG).
2. Use photos from the clinic camera if possible (native resolution is fine).

## Run inference (does not train, does not overwrite Batch 01)

From the repo root, using the same Python that has Ultralytics (Batch 01 used system Python 3.12):

```text
C:\Users\anant\AppData\Local\Programs\Python\Python312\python.exe -c "from ultralytics import YOLO; YOLO(r'models/detection/tooth_detector_v2/weights/best.pt').predict(source=r'reports/tooth_detector_v2_real_world_test/incoming', imgsz=640, conf=0.25, save=True, project=r'reports/tooth_detector_v2_real_world_test', name='predictions', exist_ok=True)"
```

Overlays will appear in `predictions/`.

Optional: also run Batch 01 `models/detection/tooth_detector_batch01/weights/best.pt` the same way into `predictions_batch01/` for a side-by-side check. Do not replace either weight file.
