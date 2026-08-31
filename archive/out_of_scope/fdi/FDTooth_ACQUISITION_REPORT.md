# Stage 2D-5 — FDTooth acquisition and inspection

Date: 2026-08-26  
Scope: legitimate access check only. No credential bypass, no fake accounts, no download of restricted files, no training, no application / ICDAS / Zenodo / `.keras` changes.

> FDI = which tooth. ICDAS = caries severity. Do not merge FDTooth with the ICDAS dataset. Do not assign ICDAS from FDI. Zenodo 14827784 remains lesion-only.

---

## Access check (this environment)

Official page: [physionet.org/content/fdtooth/1.0.0](https://physionet.org/content/fdtooth/1.0.0/) (version **1.0.0**).

The project states:

- Access policy: **only credentialed users who sign the DUA** can access files  
- License: PhysioNet Credentialed Health Data License **1.5.0**  
- DUA: PhysioNet Credentialed Health Data Use Agreement **1.5.0**  
- Required training: **CITI Data or Specimens Only Research**

Unauthenticated HEAD requests:

- `https://physionet.org/files/fdtooth/1.0.0/` → **403 Forbidden**  
- `https://physionet.org/content/fdtooth/1.0.0/files/` → **403 Forbidden**

No PhysioNet credentials are present in this project. Acquisition **stopped**.

**USER ACTION REQUIRED:** On **your own** PhysioNet account: (1) complete [credentialing](https://physionet.org/settings/profile/); (2) complete CITI **Data or Specimens Only Research** and [submit the report](https://physionet.org/about/citi-course/); (3) on the FDTooth page, **sign the project DUA**; (4) download only after PhysioNet shows you as authorized. Do not use someone else’s login. Do not share the files.

After that, place files under `data_external/fdtooth/` (already gitignored) and re-run inspection. **Do not commit images.**

---

## Download

**NOT PERFORMED.** No `data_external/fdtooth/` tree was created.

---

## Local inspection

Not possible without files.

| Item | Local result |
| --- | --- |
| JPEG count | UNKNOWN |
| Annotation count | UNKNOWN |
| Format / CSV columns | UNKNOWN |
| Exact FDI labels | UNKNOWN |
| Counts per FDI class | UNKNOWN |
| Image size / RGB | UNKNOWN |
| Box coordinates | UNKNOWN |
| Whole-tooth visual check | NOT PERFORMED |

Published page (not a local count): 241 JPEG; 1,800 boxes; 2,892 CSV tooth rows; 12 anterior teeth; F/D/N on those teeth.

---

## License / provenance (no restricted files copied)

| Field | Value |
| --- | --- |
| Name | FDTooth |
| Source | https://physionet.org/content/fdtooth/1.0.0/ |
| Version | 1.0.0 |
| Access date | 2026-08-26 (page + 403 probe only) |
| License | PhysioNet Credentialed Health Data License 1.5.0 |
| Restrictions | Credentialing + CITI + DUA |
| Permitted use | Scientific research only (documented license) |
| Redistribution | Not permitted (do not share access or files) |
| Commercial | Not permitted under “scientific research and no other” |
| Citation | Yang, Y., LI, X., Liu, K., & Elbatel, M. (2025). FDTooth … (version 1.0.0). PhysioNet. https://doi.org/10.13026/v9xk-dy61 |

---

## ICCC compatibility (from prior verified documentation, not local files)

| Question | Answer |
| --- | --- |
| RGB intraoral | YES (source; not locally opened) |
| Whole-tooth detection | YES (source; not locally opened) |
| FDI | YES (source; exact classes UNKNOWN locally) |
| 32 permanent teeth | NO |
| Anterior-only | YES |
| Auxiliary research | **PENDING** until you complete access and local inspection |
| Complete 32-tooth ICCC training set | NO |

DentalMate remains unused.

---

## What was not done

No download, no label conversion, no splits, no detector, no FastAPI/Streamlit/DB/`.keras` changes, no ICDAS merge.
