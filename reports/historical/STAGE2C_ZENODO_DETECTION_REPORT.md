# Stage 2C — Zenodo detection dataset acquisition

Date: 2026-08-26  
Scope: acquire and inspect **only** Zenodo `10.5281/zenodo.14827784`. No global search, no training, no FastAPI/Streamlit/DB changes, no `.keras` changes, no copy into `dataset/train|val|test`.

**Clinical rule:** original labels `d` / `D` are **detection classes only**. They are **not** ICDAS 0–4.

---

## 1. Source

| Field | Value |
| --- | --- |
| Title | Annotated intraoral image dataset for dental caries detection |
| DOI | [10.5281/zenodo.14827784](https://doi.org/10.5281/zenodo.14827784) |
| Record | https://zenodo.org/records/14827784 |
| Access | Open |
| License | CC BY 4.0 |
| Paper | Scientific Data [10.1038/s41597-025-05647-9](https://doi.org/10.1038/s41597-025-05647-9) |
| Claimed | 6,313 RGB intraoral images; YOLO / COCO / Pascal VOC / LabelMe |
| Classes | `d` = primary-tooth decay; `D` = permanent-tooth decay |

Canonical file: **`Dataset.zip`** (1,576,115,314 bytes).  
MD5 `89307871ac2f08f4a8f3a7da1f18db31` — **verified**.

`Benchmarking Dataset.zip` was **not** downloaded (same images plus train/val/test splits).

The verified zip was kept in local TEMP (not OneDrive) after an earlier OneDrive resume corrupted a partial download:

`C:\Users\anant\AppData\Local\Temp\zenodo_14827784_Dataset.zip`

---

## 2. Layout on disk

```
data_external/detection/raw/                    Dataset/Images (JPG)
data_external/detection/annotations/yolo/       Darknet YOLO (working format)
data_external/detection/annotations/pascal-voc/
data_external/detection/annotations/ms_coco/    train.json / valid.json / test.json
data_external/detection/archives/SOURCE.txt
data_external/detection/archives/Data_tree.txt
data_external/detection/manifest.csv
data_external/detection/manifest.jsonl
```

LabelMe JSON was not extracted (duplicate boxes). Original folder names under Images (`no_retractors`, `pilot`, `retractors` × view) were preserved.

---

## 3. Inspection (local files)

| Check | Result |
| --- | --- |
| Images found | **6,265** JPG (paper claims 6,313; **48** not present in `Dataset/Images`) |
| RGB / grayscale / X-ray-like | **6,265 RGB**; 0 grayscale; 0 X-ray-like |
| Corrupted images | **0** |
| Size range | width 617–4080; height 347–3206 |
| Duplicate filenames / content | **0** |
| YOLO / VOC files | 2,245 each |
| COCO JSON | 3 files; **2,227** listed images; **6,728** annotations |
| Images with boxes | **2,227** |
| Images with no annotation | **4,038** |
| Label files with no matching image | **36** (18 YOLO + 18 VOC, same stems) |
| Malformed annotations | **0** |
| Invalid YOLO boxes (coords outside [0,1]) | **84** |
| Empty label files | **126** |
| Class counts (VOC names) | **D 6,174**; **d 554** |
| YOLO ids | **0 → D**; **1 → d** (checked against VOC) |

COCO `train.json` / `valid.json` use category id `1=d`, `2=D`. **`test.json` inverts those ids.** Do not use COCO ids without a per-split map.

---

## 4. ICCC verdicts

| Question | Verdict |
| --- | --- |
| Usable RGB intraoral **detection** data? | **YES** (2,227 labeled RGB JPGs, lesion boxes) |
| Usable for **crops** from boxes? | **YES, with caveat** — boxes are **decay lesions**, not whole-tooth or FDI instances |
| FDI numbering? | **NO** |
| ICDAS 0–4? | **NO** |
| Train ICDAS 0–4 from this set now? | **NO** |

---

## 5. Notes for a later crop pipeline (not implemented)

- Train/detect only on the **2,227** labeled images. Unlabeled JPGs are not “healthy” or ICDAS 0.
- Crop **lesions** with padding. Do not treat a crop as a full tooth unless a later tooth-instance model says so.
- Discard or clip the **84** out-of-bounds YOLO boxes.
- Darknet: class **0 = D**, **1 = d**.
- Mixed views and retractor conditions; mixed resolutions — resize at train time.
- **Never** map `d`/`D` to ICDAS 0–4.
- **Never** copy this tree into `dataset/train`, `dataset/val`, or `dataset/test`.

---

## 6. What this stage did not do

No YOLO/TensorFlow training. No application, database, or `.keras` changes. No AlphaDent / Roboflow Prime / FDTooth download. Stage 2A/2B were not re-run.
