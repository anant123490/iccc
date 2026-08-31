# Stage 3C method note

Automatic whole-tooth boxes: **not generated**.

Reason: no local tooth-detection weights; ICDAS `.keras` models are classifiers; lesion XML is not teeth; generic YOLO/OpenCV would invent boxes.

Use CVAT / Label Studio (`annotation_project/`) for **manual** `tooth` rectangles.
