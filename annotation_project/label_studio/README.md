# Label Studio — whole-tooth rectangles

Single label: **`tooth`** (rectangle). **No FDI. No ICDAS. No lesion classes.**

## Setup (local files, no Docker required in this repo)

1. Install Label Studio however you already work (`pip install label-studio` is enough).
2. Create a project; paste `labeling_config.xml` into the labeling setup.
3. Enable **Local files** storage pointing at the repository folder that contains `fdi_detection_dataset/images/selected/`.
4. Import `tasks.json` **or** `image_list.json`.

`tasks.json` uses Label Studio local-file URLs:

`/data/local-files/?d=fdi_detection_dataset/images/selected/<filename>`

If your storage root is already `fdi_detection_dataset/images/selected`, change the prefix to `/data/local-files/?d=<filename>` instead.

5. Leave tasks unlabeled until Stage 3C.

## Export later

Export **COCO** or **YOLO** and place files under `fdi_detection_dataset/annotations/`. Do not copy Zenodo `d`/`D` files into those folders.
