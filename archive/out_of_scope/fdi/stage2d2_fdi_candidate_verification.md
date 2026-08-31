# Stage 2D-2 — Roboflow Prime FDI candidate verification

Date: 2026-08-26  
Scope: verify **only** `prime-snf1v/teeth-detection-and-numbering-agi2i`. No extra dataset search, no archive download, no training, no code or `dataset/` / Zenodo changes.

> FDI numbering and tooth detection are separate from ICDAS classification. FDI labels must never be interpreted as ICDAS labels.

> Zenodo `d/D` lesion labels remain detection-only and must never be mapped to ICDAS 0–4.

---

## 1. Candidate identity

| Field | Value |
| --- | --- |
| Listing title | teeth detection and numbering |
| Workspace / slug | `prime-snf1v` / `teeth-detection-and-numbering-agi2i` |
| Host | Roboflow Universe |
| URL | https://universe.roboflow.com/prime-snf1v/teeth-detection-and-numbering-agi2i |
| Stated type | Object Detection |
| Stated author workspace | Prime (2023) |
| Stated credits (listing) | Mohamd johamni, Noor shaker, Douaa said, George dayoob, Ali rostom |
| Trained model on listing | version **18** (hosted inference; not used here) |

---

## 2. Original source

No journal DOI or institutional dump was attached on the listing material retrieved this stage. The **original public source for this candidate is the Roboflow Universe project itself**. Live HTML was **Cloudflare-blocked**. Facts below come from:

- Indexed HTML of the **same official URL** (search capture of the Universe page).
- Public CDN files under workspace id `30zSXzIrdve0SOjB6UG1q7Munwy1` linked from that project’s sample rows (`source.roboflow.com/.../thumb.jpg` and `original.jpg`).

Those CDN objects were inspected as **listing previews only** (not a dataset zip/export).

---

## 3. Image modality

**RGB intraoral photography is not met.**

Three listing samples from this project’s CDN (filenames / alts `209.jpg`, `EL-MADANI-ISMAHANE-jpg01.jpg`, `p20191229_122513_0000.jpg`) are **grayscale panoramic radiographs (OPG)**, not camera RGB.

Higher-resolution `original.jpg` for sample `01bgnXiB5IYxdg7OJ59T` (`209.jpg`) shows extraoral panoramic geometry, L/R markers, and acquisition text including **`STD PANORAMIC`**, kV/mA/time, and **IMax-Touch 3D**.

| Question | Finding |
| --- | --- |
| RGB intraoral photographs | **No** (samples are X-ray) |
| Full-mouth intraoral photographs | **Not seen** |
| Individual-tooth RGB photographs | **Not seen** |
| Panoramic X-rays | **Yes** (verified on samples) |
| Other dental X-rays (periapical/bitewing) | **Not seen** in the three samples |
| Synthetic/rendered | **Not seen** |
| Mixed modalities | **Unknown** for the full set; **verified subset is panoramic X-ray** |

`RGB INTRAORAL: UNVERIFIED` as camera RGB. Stronger: **verified non-RGB panoramic X-ray** on inspected samples.

---

## 4. Whole-tooth bounding boxes

- Listing type: **Object Detection** (boxes, not “segmentation only” as the product type).
- Browse alts list **many FDI class names per image**, which is the usual pattern for **one box per numbered tooth**.
- Preview `thumb.jpg` / `original.jpg` **do not draw boxes**. Annotation overlay files (`annotation.png` / `annotation.jpg`) returned **404**. **No label JSON/YOLO/VOC file was downloaded.**

**Whole-tooth boxes: claimed on the listing; not file-verified.** They would still be boxes on **panoramic X-ray**, which cannot feed the RGB camera crop path.

---

## 5–6. FDI labels and exact classes

Listing text: **32 classes**; **“Teeth numbering system proposed by the World Dental Federation.”**

Sample alt-texts on this project’s browse rows include two-digit labels in the FDI adult set, for example:

- `209.jpg`: 11–18, 21–28, 31–38, 41–48  
- Other samples omit some numbers (e.g. missing 38 or 18/28 on a given image).

**Exact class list as a closed 32-set was not exported.** Verified **on sample alts**: adult FDI codes **11–18, 21–28, 31–38, 41–48** appear. Primary dentition (51–85) **not seen**. Duplicate `11` in some alts is unexplained listing noise.

FDI here would label **teeth on OPG**, not intraoral RGB.

---

## 7. Annotation format

**Listing:** Object Detection (Roboflow export would typically be YOLO/COCO/VOC after download).  
**This stage:** format **NOT VERIFIED** (no annotation file opened).

---

## 8–9. Image and annotation counts

| Quantity | Status |
| --- | --- |
| Images | Listing display **“1.4k images”** — rounded UI, **not an exact integer**. Exact count: **NOT VERIFIED** |
| Annotations | **NOT VERIFIED** |
| Classes | Listing: **32** |
| Train / val / test | **NOT VERIFIED** |
| Other listing stats (indexed) | ~5k views, **199 downloads**, updated ~2 years ago |

---

## 10. License

Indexed official listing Cite block links **[MIT](https://choosealicense.com/licenses/mit/)** next to the BibTeX for this slug.

Live page was not loaded (Cloudflare), so the license widget was not re-read in-browser.

| Question | Finding |
| --- | --- |
| License name | **MIT** on indexed Universe listing |
| Commercial use (MIT text) | Permitted (with copyright notice) **if** that listing is accurate |
| Attribution | MIT notice; listing also asks for BibTeX citation |
| Extra Roboflow terms | Universe **Download** still sits on Roboflow’s platform ToS/account rules. Those terms were **not** independently quoted this stage |

`LICENSE: VERIFIED` at listing-index level as **MIT**, with **platform ToS not fully quoted**.

---

## 11. Accessibility

| Path | Finding |
| --- | --- |
| Public listing URL | Exists; **Cloudflare** blocks automated/browser-bot loads |
| Unauthenticated full dump | **Not demonstrated** |
| Roboflow account / Download Project | Listing shows **Download Project** and 199 downloads — **account-required download is the expected path** (no credentials used) |
| Public API without key | **Not used / not verified** |
| This stage | **Did not download** the dataset |

---

## 12. RGB verification evidence

1. Live Universe HTML: Cloudflare challenge only.  
2. Official URL indexed snippets: Object Detection, 32 WDF classes, 1.4k images, MIT cite.  
3. CDN `original.jpg` for `209.jpg`: **STD PANORAMIC** radiograph, not an intraoral color photo.  
4. Two further project thumbs: same OPG appearance.

Title “teeth detection” is **not** evidence of RGB intraoral photos.

---

## 13. ICCC compatibility

Intended path: RGB intraoral camera → whole-tooth → FDI → tooth crop → ICDAS 0–4.

| Requirement | Required | Result |
| --- | --- | --- |
| RGB | YES | **Fail** (OPG samples) |
| Intraoral photography | YES | **Fail** |
| Whole-tooth bounding boxes | YES | **Unverified files**; listing claims OD |
| FDI numbering | YES | **Listing + sample class names** (on X-ray) |
| FDI classes documented | YES | **32 WDF claimed**; 11–48 seen in alts |
| Legally usable license | YES | **MIT on listing index** |
| Public/downloadable | Preferred | **Likely with Roboflow account; not proven** |
| Suitable for camera pipeline | YES | **No** — wrong modality |

### `FDI DATASET: NOT READY`

Essential RGB intraoral photography is **not** satisfied. Do not use this set to train the camera tooth/FDI stage.

---

## 14. Unknown / unverified fields

- Exact image/annotation/split counts  
- Annotation file format and box quality  
- Whether **every** image is OPG (only samples checked; those samples **are** OPG)  
- Live license widget (index vs Cloudflare)  
- Unauthenticated download  
- Original clinic/paper behind the Universe upload  
- Patient identifiers (one filename looks like a personal name — privacy risk if exported)

---

## 15. Final recommendation

**Reject this Roboflow Prime project for the ICCC camera pipeline.** It is a **panoramic X-ray FDI detection listing**, not RGB intraoral whole-tooth+FDI data.

Keep it logically separate from `dataset/` (ICDAS) and `data_external/detection/` (Zenodo lesions). Do not download it for ICDAS or camera FDI work. Do not map its 11–48 classes to ICDAS 0–4.

This stage does **not** name or fetch a replacement (per 2D-2 stop rule).
