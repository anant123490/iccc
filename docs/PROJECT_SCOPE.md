# Project scope (active)

FDI tooth numbering is **out of scope**. Do not create FDI labels, train an FDI model, search for FDI datasets, or make FDI an intermediate step.

## Active pipeline

```text
Patient Registration
  → Visit Creation
  → RGB Intraoral Photo
  → Whole-Tooth Detection (class 0 = tooth)
  → Individual Tooth Cropping
  → ICDAS 0–4 Classification
  → Grad-CAM
  → Clinical Report
```

## In scope

- ICDAS grades **0, 1, 2, 3, 4** only (five classes).
- Whole-tooth boxes (no lesion `d`/`D` as teeth).
- Reuse of existing detector, crops, and MobileNetV3+CBAM training code.

## Out of scope

- FDI numbering.
- ICDAS 5 and 6 (do not remap to 4).
- Treating YOLO crops or auto-ICDAS CSV as dentist ground truth.

## Historical paths to keep

`fdi_detection_dataset/` is a **folder name only** (RGB intraoral photos + tooth detector data). It is not an FDI labeling project. Do not delete detector weights, Batch 01 labels, or `data/tooth_crops/generated/` just because FDI is out of scope.
