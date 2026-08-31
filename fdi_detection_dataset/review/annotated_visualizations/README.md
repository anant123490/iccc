# Annotated visualizations (Stage 3C)

No overlay images were written.

Stage 3C did **not** generate whole-tooth boxes (no reliable local tooth detector; fabricating boxes is forbidden).

Original selected photos were **not** copied here.

After humans annotate in Stage 3C/3D, run:

```text
python tools/visualize_annotations.py --format yolo --images fdi_detection_dataset/images/selected --labels fdi_detection_dataset/annotations/yolo --out fdi_detection_dataset/review/annotated_visualizations
```
