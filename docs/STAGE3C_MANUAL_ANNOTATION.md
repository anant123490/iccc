# Stage 3C — Manual whole-tooth annotation

**Status:** pretrained-detector search is **closed**. Boxes will be **HUMAN-GENERATED**, not model-generated.

Verified 2026-08-26:

| Check | Result |
| --- | --- |
| `fdi_detection_dataset/images/selected/` | **420** JPG |
| Class | **exactly** `0` = `tooth` (`classes.yaml`, `annotation_project/cvat/labels.json`) |
| YOLO placeholders | **420** empty `.txt` — **not** overwritten |
| ICDAS `dataset/train` | still absent; `dataset/annotations.csv` not touched |
| Batches | **7 × 60** (`annotation_batches/Batch_01` … `Batch_07`) |

**Do not** draw FDI, ICDAS, lesions, gums, lips, or retractors. **Do not** import Zenodo `d`/`D` XML.

Progress file (update after each session):

`annotation_project/annotation_progress.csv`

Columns: `filename`, `batch`, `annotation_status`, `number_of_tooth_boxes`, `reviewer_status`.

---

## How you should start Batch_01 (CVAT — preferred)

This repo does **not** run a CVAT server. Use [CVAT online](https://app.cvat.ai/) (free account) or any CVAT you already have.

### 1. Open CVAT

In a browser go to: **https://app.cvat.ai/**  
Sign in.

### 2. Create the project (once)

1. **Projects → Create new project**  
   Name: `iccc_whole_tooth_detection`
2. Add **one** label:
   - Name: `tooth`
   - Type: **Rectangle**
   - No attributes, no extra classes, no 11–48
3. You can copy `annotation_project/cvat/labels.json` if the UI accepts JSON labels.

### 3. Create a **task** for Batch_01 only (60 images)

Do **not** upload all 420 on the first task.

1. **Tasks → Create new task**
2. Name: `iccc_batch_01`
3. Assign it to project `iccc_whole_tooth_detection`
4. **Source: My computer** (local files)
5. Upload **only** the 60 files listed in:

`annotation_batches/Batch_01/cvat_upload_filenames.txt`

Those files live in:

`fdi_detection_dataset/images/selected\`

**Windows Explorer:** open that folder, sort by name, and upload the filenames from `cvat_upload_filenames.txt` (search/paste names, or multi-select matching files). Do **not** pick files from `images/review/` or `data_external/`.

6. Leave **annotations empty**. Do **not** attach YOLO/COCO/VOC/lesion XML.
7. Submit / create the task.

### 4. Draw boxes

1. Open task `iccc_batch_01` → **Job**
2. Tool: **Rectangle**
3. Label: **`tooth`** only  
4. One box per **visible tooth crown** (see `TOOTH_ANNOTATION_GUIDELINES.md`)
5. Save often (CVAT save / Ctrl+S depending on UI)

### 5. After Batch_01 (later — not this step)

Export **YOLO 1.1** or **COCO 1.0**. A later stage will copy them into `fdi_detection_dataset/annotations/` without mixing lesion files. Then set `annotation_status` to `done` for those 60 rows in `annotation_progress.csv`.

---

## How you should start Batch_02 (CVAT)

Batch_02 is **60 images** chosen from the 360 remaining after Batch_01. Selection used YOLO candidate stats (count extremes, low confidence, overlap) plus view/patient diversity. **Do not import YOLO boxes as labels.**

1. **Tasks → Create new task** named `iccc_batch_02` in project `iccc_whole_tooth_detection`.
2. Upload **only** the 60 JPGs in `annotation_batches/Batch_02/seed_60/` (copies; `images/selected/` was not modified).
3. Name list: `annotation_batches/Batch_02/cvat_upload_filenames.txt`.
4. Leave annotations **empty**. Draw `tooth` rectangles from scratch, same rules as Batch_01.
5. Optional: open `annotation_batches/Batch_02/yolo_overlays_for_review/` in another window as a checklist. Those overlays are **wrong in places** (duplicate boxes, gums).
6. Details: `annotation_batches/Batch_02/README.md` and `reports/BATCH_02_SELECTION.md`.

---

## Label Studio (only if you already use it)

Do **not** install new packages unless you choose this path yourself.

1. Start Label Studio the way you already do.  
2. New project; paste `annotation_project/label_studio/labeling_config.xml` (single `tooth` rectangle).  
3. Local storage root = folder containing `fdi_detection_dataset/images/selected/`.  
4. Import **only Batch_01** filenames (filter `annotation_batches/Batch_01/image_list.csv`), not the full `tasks.json` 420 list, for the first session.

---

## What not to do

No model downloads, no FDI, no ICDAS, no YOLO training, no edits under `dataset/`, `ml/`, `models/`, backend, frontend, or original Zenodo trees.
