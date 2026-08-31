# Stage 2D-4 — Final verification of RGB intraoral + whole-tooth + FDI candidates

Date: 2026-08-26  
Scope: verification only. No Cloudflare bypass, no credentials, no downloads, no training, no application / `dataset/` / Zenodo / `.keras` changes.

Architecture **unchanged**: RGB intraoral → whole-tooth detection → **FDI** numbering → tooth crop → ICDAS 0–4.

> FDI = which tooth. ICDAS = caries severity. Never map FDI, `d`/`D`, F/D/N, or Black classes to ICDAS. Zenodo 14827784 stays **lesion-only**.

---

## 1. Stage objective

Decide whether a **realistically usable** public dataset exists for the missing tooth-detection + FDI stage.

## 2. ICCC target pipeline

RGB intraoral photograph → whole-tooth detection → FDI tooth number → tooth crop → ICDAS 0–4 classifier.

Preferred: full-mouth, many teeth per image, 32 permanent FDI classes, open download. Lack of 32 classes does **not** change the architecture.

## 3. Candidates investigated

1. **DentalMate** (Roboflow workspace DentalMate6v) — first priority  
2. **FDTooth** (PhysioNet) — second priority  
3. Related literature already tied to this search (SegmentAnyTooth only; no new broad search)

---

## 4. DentalMate verification

| Field | Result |
| --- | --- |
| Original paper | **NONE FOUND** |
| GitHub / institutional repo | **NONE FOUND** |
| Zenodo / Figshare / Mendeley mirror | **NONE FOUND** |
| Official documentation beyond Roboflow | **NONE FOUND** |
| Public host | Roboflow Universe [dentalmate6v](https://universe.roboflow.com/dentalmate6v) |

Workspace **index** (not sample pixels):

- Intraoral Tooth Numbering FDI — **1.21k** images, **32** classes  
- Front / Upper / Lower Intraoral Tooth Numbering — 408 / 407 / 387  
- Intraoral Tooth Detection — 1.21k  
- **dentalmate-dataset 2** — 3.49k; indexed detector classes **caries, fractured-teeth, gingivitis, misaligned-teeth, multiple-tooth-loss** — **not FDI**

Project HTML remains Cloudflare-blocked. **No sample JPEG inspected.** After Prime (title vs panoramic X-ray), titles are not enough.

**FDI 32-class status: UNVERIFIED**  
**Not a strong candidate.**

| Check | Status |
| --- | --- |
| RGB intraoral | UNVERIFIED |
| Whole-tooth annotation | UNVERIFIED |
| FDI numbering | UNVERIFIED |
| Public source | VERIFIED (workspace URL) |
| License | UNVERIFIED |
| Access | UNKNOWN |
| Commercial use | UNKNOWN |
| Redistribution | UNKNOWN |

---

## 5. FDTooth verification

| Field | Result |
| --- | --- |
| Source | [physionet.org/content/fdtooth/1.0.0](https://physionet.org/content/fdtooth/1.0.0/) DOI [10.13026/v9xk-dy61](https://doi.org/10.13026/v9xk-dy61) |
| Paper | Scientific Data [10.1038/s41597-025-05348-3](https://doi.org/10.1038/s41597-025-05348-3) |
| Intraoral images | **241** JPEG, 5760×3840 |
| CBCT | **241** DICOM — **not** for the camera classifier |
| Whole-tooth boxes | **1,800** on photographs (MakeSense JSON) |
| Tooth-level labels | **2,892** CSV rows (F / D / N) |
| Patients | 241; one photo + one CBCT each |

**RGB:** VERIFIED — paper Fig. 3(a) intraoral photograph; Fig. 5 labelled intraoral images with rectangular boxes. Files not downloaded.

**Whole-tooth:** VERIFIED at source — boxes on anterior **teeth**, not caries lesions. Box count (1,800) ≠ 241×12 (2,892), so **complete per-tooth box coverage is UNVERIFIED** without JSON.

**FDI:** VERIFIED — CSV uses **FDI World Dental Federation / International Dental Federation two-digit** numbers for **12 anterior teeth** (fully erupted **incisors and canines**, both arches). Exact exported codes (file contents) **UNVERIFIED**. Posterior 14–18 / 24–28 / 34–38 / 44–48 **out of scope**. JSON visualisation is FD vs no-FD colour; FDI lives in the **CSV**, not as a 32-class YOLO pack.

**Access:** CREDENTIALLED — PhysioNet credentialing (typically **no download fee**), **CITI Data or Specimens Only Research**, sign **DUA 1.5.0**.

**License:** VERIFIED — PhysioNet Credentialed Health Data License **1.5.0**.

**Commercial use:** **NOT ALLOWED** — “use the data for the sole purpose of lawful use in **scientific research and no other**.”

**Redistribution:** **NOT ALLOWED** — “will **not share access** to PhysioNet restricted data with anyone else”; FAQ: each person must credential individually.

### FDTooth special decision: **B**

- **Technically compatible:** yes (RGB photos + whole-tooth boxes + explicit FDI).  
- **Practically suitable for the intended full ICCC camera pipeline:** **no** — anterior-only, credentialed CITI/DUA, research-only, no file sharing, not 32-class.

Use as **research/validation** after legal access, not as an open production 32-tooth detector dump.

---

## 6. Other serious alternatives (from related papers only)

**SegmentAnyTooth** (JDS; GitHub [thangngoc89/SegmentAnyTooth](https://github.com/thangngoc89/SegmentAnyTooth)): 5,000 RGB intraoral photos + FDI + YOLO boxes **in the paper**. Images **not released**. Weights non-commercial by email. Not a DentalMate/FDTooth mirror.

No other new public dump from these sources.

---

## 7–17. Comparison

| Item | DentalMate | FDTooth |
| --- | --- | --- |
| RGB intraoral | UNVERIFIED | VERIFIED |
| Whole-tooth | UNVERIFIED | VERIFIED |
| FDI | UNVERIFIED | VERIFIED |
| Exact FDI classes | UNVERIFIED | 12 anterior (incisors+canines); numeric list UNVERIFIED |
| Images | listing 1.21k | 241 photos |
| Annotations | UNVERIFIED | 1800 boxes; 2892 CSV |
| Format | UNVERIFIED | JPEG + CSV + JSON |
| Public source | Roboflow only | PhysioNet + Scientific Data |
| License | UNVERIFIED | VERIFIED (Credentialed 1.5.0) |
| Access | UNKNOWN | CREDENTIALLED |
| Commercial | UNKNOWN | NOT ALLOWED |
| Redistribution | UNKNOWN | NOT ALLOWED |
| 32-class FDI | UNVERIFIED | NO |

---

## 18–20. ICCC compatibility

| | |
| --- | --- |
| Technical suitability (FDTooth) | **PARTIAL** — can train/validate **anterior** tooth instance + FDI after DUA |
| Practical suitability (full camera 32-tooth product) | **LOW** |
| DentalMate | **UNKNOWN** until samples and license are seen |
| Open 32-class RGB FDI dump | **NOT FOUND** this stage |

Do **not** replace FDI with another numbering system. Do **not** merge FDI and ICDAS.

---

## 21. Remaining uncertainty

- DentalMate pixel modality and real class names  
- FDTooth JSON–CSV join and missing boxes vs 12 teeth  
- Exact 12 FDI integers in the CSV  
- Product use of models trained on PhysioNet data (beyond “research only”) — legal review, not guessed here  

---

## 22. Final recommendation

**B. TECHNICALLY COMPATIBLE DATASET FOUND BUT ACCESS/COVERAGE LIMITATIONS REMAIN**

Best candidate: **FDTooth**. If ICCC needs a **research** anterior detect→FDI→crop prototype, credential PhysioNet later (not this stage). If ICCC needs an **open 32-class camera** set, **none is verified**. Do not treat DentalMate as ready. Do not download now. Do not implement detectors.

---

**Final decision: B**
