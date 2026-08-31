# ICDAS v2 working tree

Public images stay in `data_external/detection/` (not copied here).

User copies go to `user_images/` (originals elsewhere are not modified).

Tooth crops (when a dentist or a **whole-tooth** detector produces them) go in `crops/`.

**Do not** fill `crops/` from lesion `d`/`D` boxes.

Labels: `manifest/icdas_labels.csv` (project-generated ICDAS only).

Built splits: `final/train|val|test/0-4` — separate from existing `dataset/`.
