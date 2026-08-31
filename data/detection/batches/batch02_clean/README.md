# Batch 02 CLEAN

KEEP-only unique-stem copy for **future** YOLO tooth-detector training. **Not trained in this step.**

| Path | Meaning |
|------|---------|
| `images/{train,valid,test}/` | Retained JPGs |
| `labels/{train,valid,test}/` | Matching YOLO rectangles `0 xc yc w h` |
| `data.yaml` | Ultralytics layout |
| `held_out/review/` | Near-gray / gallery / screenshot representatives |
| `held_out/excluded/` | Roboflow augmentation duplicates |
| `file_classification.csv` | KEEP / REVIEW / EXCLUDE for all 2907 source files |

Do not modify `batch02/yolo_detection/` or Batch 01.
