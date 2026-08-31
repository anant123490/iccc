# Stage 2B — Global public dataset search

Date: 2026-08-26  
Scope: internet-wide search beyond Kaggle/Hugging Face. Original sources checked. **No application changes, no training, no patient/visit creation, no ICDAS invention, no large unverified downloads, no overwrites of models.**

Clinical rules applied throughout:

- Do not convert caries / healthy–diseased / mild–moderate–severe / initial–moderate–extensive to ICDAS unless the **original paper/dataset defines the mapping**.
- ICDAS II is **not** automatically ICDAS 0–4.
- Black’s classification is **not** ICDAS.
- Panoramic/X-ray sets are **not** camera RGB training data.

---

## 1. BEST ICDAS DATASETS

No publicly downloadable dump was found whose **original source** proves clinician ICDAS **0–4 only** on intraoral RGB photos. The strongest clinical sets are **0–6 (D0–D6)** and **not posted**.

### Rank 1 — HI Bogi / BMC Oral Health (best clinical match)

| Field | Finding |
| --- | --- |
| Name | Intraoral ICDAS photographs used to train YOLOv8x for the HI Bogi application |
| Original source | [BMC Oral Health, DOI 10.1186/s12903-025-07486-x](https://bmcoralhealth.biomedcentral.com/articles/10.1186/s12903-025-07486-x) |
| Direct download | **None** |
| Paper | Same URL |
| License | Paper open access. Images: **“Data are available from the corresponding authors.”** |
| Images | **3,221 JPG** (train 2,266 / val 635 / test 320), 640×640 for training |
| Modality | **RGB intraoral photographs** |
| Annotations | Roboflow **bounding boxes** tied to ICDAS labels |
| Exact labels | **D0, D1, D2, D3, D4, D5, D6** |
| Genuinely ICDAS? | **Yes** (ICDAS protocol, dentists, two ICDAS experts; dual-expert agreement required) |
| Grades | D0–D6 |
| Tooth boxes | Yes |
| FDI | No |
| Patient IDs | Not public |
| Access | **Author request** |
| Commercial/academic | Unknown until authors license the files |
| Downloadable now? | **No** — PAPER-FOUND / DATASET-NOT-PUBLIC |
| ICCC | Best future source. If obtained: train **0–4 only**; **exclude 5/6**; **do not remap 5/6 → 4**. |

### Rank 2 — IDCCD (Indonesian Dental Caries Characteristic Dataset)

| Field | Finding |
| --- | --- |
| Original source | [IJEECS, DOI 10.11591/ijeecs.v38.i1.pp381-392](https://doi.org/10.11591/ijeecs.v38.i1.pp381-392) |
| Direct download | **None found** |
| Labels | **D0–D6** ICDAS, dental annotators, detection models (YOLOv8 / DETR / Faster R-CNN) |
| Modality | Primary **RGB** photos (Indonesian collection) |
| Boxes | Yes |
| FDI | No |
| Status | PAPER-FOUND / DATASET-NOT-PUBLIC |
| ICCC | Same research line as HI Bogi (HI Bogi cites IDCCD). Request authors; do not assume a public Roboflow dump. |

### Rank 3 — CGD-Det pediatric smartphone ICDAS 0–6

| Field | Finding |
| --- | --- |
| Original source | Hou et al., *Biomedical Signal Processing and Control* (2026), [DOI 10.1016/j.bspc.2026.110557](https://doi.org/10.1016/j.bspc.2026.110557) |
| Images | 1,241 clinical/community photos; **6,260** RGB after augmentation |
| Labels | ICDAS **0–6**, expert consensus (abstract) |
| Modality | **Smartphone RGB** (good camera analog) |
| Boxes | Yes (YOLO-style detector) |
| Status | PAPER-FOUND / DATASET-NOT-PUBLIC |
| ICCC | Promising if released. Do not treat augmented copies as independent patients. |

### Rank 4 — Odontify Clean ICDAS Dataset V2 (only public *claimed* 0–6 photos)

| Field | Finding |
| --- | --- |
| Listing | [Kaggle: leonardoaranguiz/odontify-clean-icdas-dataset-v2](https://www.kaggle.com/datasets/leonardoaranguiz/odontify-clean-icdas-dataset-v2) |
| Original gold-standard paper | **Not found** |
| License (listing) | **CC BY-NC-SA** (non-commercial, share-alike) |
| Images | ~**3,010** files; folders **0–6** parsed from **filenames** |
| Genuinely ICDAS? | **Not verified.** Filenames are not a clinician protocol. |
| Boxes / FDI | No |
| Access | Kaggle account |
| Downloadable? | **Yes** (not downloaded this stage) |
| ICCC | Weakest “ICDAS” among ranked sets. Usable later only after **dentist review**. Keep 5/6 out of 0–4. NC license likely blocks commercial ICCC. |

### Rank 5 — Research Square ICDAS 0 vs 2 occlusal photos

[DOI 10.21203/rs.3.rs-3125352/v1](https://doi.org/10.21203/rs.3.rs-3125352/v1) — private clinical, **only grades 0 and 2**. Too narrow. PAPER-FOUND / DATASET-NOT-PUBLIC.

**ICDAS 0–4 specifically:** no public clinician-verified RGB set. Closest path is author-request D0–D6 then **drop D5/D6**.

---

## 2. BEST DETECTION / FDI DATASETS

### Rank 1 — Aga Khan University / Scientific Data (best **open** RGB detection)

| Field | Finding |
| --- | --- |
| Name | Annotated intraoral image dataset for dental caries detection |
| Original source | Zenodo **[10.5281/zenodo.14827784](https://zenodo.org/records/14827784)** (open; CC BY 4.0 per Zenodo API). Paper: [Scientific Data 10.1038/s41597-025-05647-9](https://www.nature.com/articles/s41597-025-05647-9) |
| Images | **6,313** JPG |
| Modality | **RGB intraoral** (retractor / no retractor / pilot; five standard views) |
| Annotations | LabelMe, YOLO, VOC, COCO |
| Exact labels | **`d`** primary decay, **`D`** permanent decay |
| Genuinely ICDAS? | **No** — do not map `d`/`D` to ICDAS |
| Boxes | Yes |
| FDI | No |
| License | **CC BY 4.0**, `access_right: open` |
| Size | ~1.57 GB + ~1.58 GB zips — **not downloaded** in 2B |
| ICCC | Best open set for a later **detect → crop** pipeline, then separate ICDAS labeling. |

### Rank 2 — AlphaDent (open RGB pathology / instance seg)

| Field | Finding |
| --- | --- |
| Sources | [Zenodo 16582489](https://zenodo.org/records/16582489), [GitHub ZFTurbo/AlphaDent](https://github.com/ZFTurbo/AlphaDent) (Apache 2.0), Hugging Face `ZFTurbo/AlphaDent` |
| Images | ~**1,320** DSLR RGB; **patient IDs** (`p001_…`) |
| Labels | Pathology including **Black caries 1–6 (location)** — **not ICDAS** |
| Boxes/masks | Yes (YOLO instance seg) |
| FDI | No |
| Size | ~4.9 GB — **not downloaded** |
| ICCC | Detection/seg experiments only. Never train ICDAS on Black classes. |

### Rank 3 — Roboflow “teeth detection and numbering” (Prime) — claimed **boxes + 32 FDI**

| Field | Finding |
| --- | --- |
| Page | [universe.roboflow.com/prime-snf1v/teeth-detection-and-numbering-agi2i](https://universe.roboflow.com/prime-snf1v/teeth-detection-and-numbering-agi2i) |
| License (listing) | **MIT** |
| Classes | **32**, World Dental Federation numbering |
| Automated fetch | **Cloudflare** blocked full page; image count **not re-verified** this session |
| ICCC | Strongest **public listing** for category F **if** samples are intraoral RGB (must confirm; many dental Roboflow sets are panoramic). Account/API required. Not downloaded. |

### Rank 4 — FDTooth (verified RGB **boxes + FDI**, restricted)

| Field | Finding |
| --- | --- |
| Source | [PhysioNet fdtooth 1.0.0](https://physionet.org/content/fdtooth/1.0.0/) DOI [10.13026/v9xk-dy61](https://doi.org/10.13026/v9xk-dy61) |
| Images | **241** intraoral JPEG (5760×3840) + CBCT; **1,800** boxes; **12 anterior teeth** |
| Labels | **F / D / N** (fenestration / dehiscence / normal) — **not caries, not ICDAS** |
| FDI | **Yes** (CSV) |
| Patient IDs | **Yes** (001–241) |
| Access | Credentialed PhysioNet + **CITI** + **DUA** |
| ICCC | Numbering/crop research only. Do not train the camera classifier on CBCT. |

### Rank 5 — SegmentAnyTooth (best clinical protocol, images **not public**)

[DOI 10.1016/j.jds.2025.01.003](https://doi.org/10.1016/j.jds.2025.01.003): **5,000** intraoral photos, five views, **FDI + surfaces**, YOLO boxes. Code may be MIT; **images not released**. PAPER-FOUND / DATASET-NOT-PUBLIC.

### Rank 6 — IO150K / Teeth-SEG

[Project page](https://zoubo9034.github.io/TeethSEG/). Claims 150k “open” images but **~80k renders + ~70k plaster + ~0.8k real RGB**. [github.com/zoubo9034/TeethSEG](https://github.com/zoubo9034/TeethSEG) returned **404**. Not usable as camera ICDAS data.

**Category F (boxes AND FDI) summary:** no large **easy-open** RGB dump was fully file-verified. Candidates: Roboflow Prime (listing), FDTooth (credentialed, anterior only), Yoon 24,578 and SegmentAnyTooth (not public).

---

## 3. DATASETS REJECTED AND WHY

| Dataset | Why rejected for ICCC camera ICDAS 0–4 |
| --- | --- |
| **Yoon et al. 24,578** intraoral photos ([J Dent 104821](https://doi.org/10.1016/j.jdent.2023.104821)) | **Not public.** Has tooth numbers + caries **stages 1–2–3** (not ICDAS). Scientific Data paper states it is not open source. |
| **Kühnisch et al.** 2,417 tooth photos ([JDR](https://doi.org/10.1177/00220345211032524)) | **Not public.** Labels: caries-free / noncavitated / cavitation. **Do not convert to ICDAS.** |
| **Roboflow ICDAS II** Healthy / Initial / Moderate / Extensive | Merged **ICDAS II buckets**, not 0–4. **No mapping.** Not re-downloaded. |
| **AlphaDent as ICDAS** | Black **location** classes ≠ ICDAS severity. |
| **Korean intraoral studies using ICDAS 4–6 vs not** | Binary collapse. **No 0–4 mapping.** |
| **Child-OID** 1,368 images | GitHub: dataset “will be released soon.” Labels normal/abnormal, not ICDAS grades. |
| **UFBA-425 Figshare**, **Roboflow panoramic FDI**, **DENTEX** | **X-ray / panoramic** — wrong modality for the RGB camera. |
| **3D ICDAS STL / phantom CNN repos** | Not intraoral RGB photographs. |
| **Mendeley `5vb5tvkjb5`** | **Dataset Not Found** on the original Mendeley URL at verification time. |
| **Local `dataset/` + `annotations.csv`** | Not an internet corpus. Stage 2A: **16** excluded 5/6 images; **643** CSV rows with **0** files on disk. |
| **IO150K renders/plaster** | Not patient camera RGB at scale. |

---

## What this means for ICCC

- **Training data is not ready.** Genuine ICDAS RGB is behind author request; the only public 0–6 photo dump is **filename-based and unverified**.
- Open RGB **detection** exists (Zenodo 6,313; AlphaDent) and is the right later path for cropping, **not** for softmax 0–4 labels.
- Open RGB **FDI + boxes** is still thin: PhysioNet (hard access, anterior F/D/N) or Roboflow (must confirm photos vs X-rays).

---

BEST ICDAS DATASET: HI Bogi / BMC Oral Health ICDAS D0–D6 (author request); public claimed alternative Odontify Clean ICDAS V2 (filename 0–6, unverified)  
ICDAS GRADES AVAILABLE: D0–D6 in HI Bogi/IDCCD/CGD-Det papers; none clinician-verified on disk  
PUBLIC DOWNLOAD: NO for verified clinician ICDAS 0–4; YES for Odontify listing (unverified) and for detection sets (Zenodo 14827784, AlphaDent)  
LICENSE: HI Bogi images not posted; Odontify listing CC BY-NC-SA; Zenodo 14827784 CC BY 4.0; AlphaDent Apache 2.0; FDTooth PhysioNet credentialed DUA  
BEST DETECTION DATASET: Annotated intraoral caries detection, Zenodo 10.5281/zenodo.14827784 (6,313 RGB, labels d/D)  
FDI AVAILABLE: Not in a large easy-open RGB dump; Roboflow Prime 32-class listing (verify RGB); FDTooth anterior FDI (PhysioNet); SegmentAnyTooth/Yoon not public  
TRAINING DATA READY: NO  
RECOMMENDED NEXT ACTION: Request HI Bogi/IDCCD from corresponding authors for genuine ICDAS D0–D6 RGB+boxes; optionally later download Odontify V2 only for clinician filename audit (exclude 5/6, no remap) and/or Zenodo 14827784 for detect-crop (never map d/D to ICDAS). Do not train. Copy ordinal `.keras` files before any future training. Do not map ICDAS-II 4-buckets, Black classes, or binary caries to ICDAS 0–4.
