# GitHub YOLO tooth-detector audit (KOUSHIK-9)

Date: 2026-08-26  
Repo: https://github.com/KOUSHIK-9/Tooth-Detection-Using-Yolo  
**No weights downloaded. No packages installed. No 5-image inference. No ICCC files modified except this folder.**

Verdict on committed `yolo11n.pt`: **A) standard Ultralytics YOLO11n base weights — not a tooth-trained detector.**

Overall candidate: **REJECTED / NOT READY** as a SegmentAnyTooth alternative.

---

## Weight verification: `yolo11n.pt`

| Check | Finding |
| --- | --- |
| In GitHub tree | Yes (5,613,764 bytes) |
| Locally on this machine | **WEIGHT_NOT_LOCALLY_AVAILABLE** (not inspected; not downloaded) |
| Typical YOLO11n COCO size | ~5.6 MB — **matches** committed file |
| Ultralytics docs | `YOLO("yolo11n.pt")` = **COCO-pretrained** YOLO11n |
| README step 4 | “Download a pretrained YOLOv8 model” / `YOLO("yolov8n.pt")` — same pattern |
| Also committed | `yolov8s.pt` (~22.6 MB) — typical **stock YOLOv8s** size |
| Tooth-trained `best.pt` | Notebook wrote `runs/detect/train4/weights/best.pt` (**16.6 MB**, from `yolov10s.pt` train). **Not in the Git tree** |
| `Train/YOLO12m/` | Curves + val plot JPEGs only — **no `.pt`** |

**Conclusion:** Do **not** treat repo `yolo11n.pt` as a tooth detector. Using it on the 420 photos would be **generic COCO** detection (already forbidden).

---

## Answers 1–20

1. **Is yolo11n.pt a trained tooth detector?** **No** (stock nano checkpoint by size + naming + docs).  
2. **Merely standard YOLO11n?** **Yes** — that is the supported interpretation.  
3. **Dataset (for the *notebook* training, not for yolo11n.pt):** `data.yaml` + Colab logs: `dental_dataset` with **496 train / 201 val** images scanned; README **~500** annotated intraoral with FDI. Dataset folder **not** in the repo. Filenames include Roboflow-style `cate*-*.rf.*` and camera-like timestamps.  
4. **RGB intraoral?** **Likely** for the *training data described* (README + FDI tooth names + JPEG names). Not verified by opening pixels here. **Not** what `yolo11n.pt` was trained on (COCO).  
5. **Training images:** README ~500; notebook train scan **496**.  
6. **Classes:** 32 names in `data.yaml` (Canine/Incisor/Molar + FDI in parentheses).  
7. **Generic tooth vs FDI:** **FDI-style numbering** in class names, not a single `tooth` class.  
8. **Checkpoint in repo?** `yolo11n.pt` and `yolov8s.pt` yes; **tooth `best.pt` no**.  
9. **Downloadable without agreement?** Git blobs are public **but downloading was not done**. Stock Ultralytics files are public; they are still **not tooth weights**.  
10. **Academic use of yolo11n.pt?** COCO YOLO11n is typically **AGPL-3.0**; still **wrong task**. Tooth `best.pt` license **UNKNOWN** (unpublished).  
11. **Repo license:** GitHub API `"license": null`. **LICENSE_STATUS = UNKNOWN**. Empty `README.txt`.  
12. **Trained-weight license:** **UNKNOWN** (weights not published). Stock files inherit **Ultralytics AGPL**.  
13. **Inference script:** No dedicated `.py`. README shows generic `model('image.jpg')`. Notebook: `model.predict(source="dental_dataset/images/test", ...)`.  
14. **Image size:** **640** (`imgsz=640` in README and notebook train).  
15. **YOLO version:** Mixed: README YOLOv8/11; notebook train **YOLOv10s**; plots folder **YOLO12m**. Committed bases: **YOLO11n + YOLOv8s**.  
16. **Whole teeth?** Intended by `data.yaml` (whole-tooth FDI classes). **Not** what `yolo11n.pt` does.  
17. **Individual teeth?** Intended 32 instance classes.  
18. **Intraoral vs X-ray?** README says intraoral. Not X-ray in the description.  
19. **Bounding boxes?** YOLO detect = boxes. Notebook prints many boxes per test image.  
20. **Restrictions:** Unknown repo license; AGPL on Ultralytics bases; **do not assume** custom weights may be redistributed even if obtained later.

---

## Compare with ICCC (420 images, class `0 = tooth`)

If a **tooth-trained** checkpoint existed, FDI class IDs **could be remapped** to `0 = tooth` (boxes kept, FDI discarded). **Do not convert now.** There is **no** such checkpoint in the repo.

---

## 5-image test

**Not run.** `WEIGHT_NOT_LOCALLY_AVAILABLE`. Would not install or download. Testing stock `yolo11n.pt` would not answer the tooth-detector question.

Intended sample listed in `test_images.csv` (5 views, one each).

---

## Final decision questions

| # | Answer |
| --- | --- |
| 1. Tooth-trained `yolo11n.pt`? | **No** |
| 2. Trained on RGB intraoral? | **`yolo11n.pt`: no (COCO). Dataset in yaml/notebook: likely yes, unpublished.** |
| 3. Whole teeth detected? | **Not by committed nano weights** |
| 4. Boxes available? | YOLO format **in principle**; no usable tooth weights |
| 5. Without license agreement? | Stock files: no SAT-style email; **AGPL / unknown repo**. Tooth `best.pt` missing. |
| 6. Weight license clear? | **No** for custom; AGPL for stock |
| 7. Tested locally? | **No** |
| 8. Suitable for 420 images? | **Not with committed weights** |
| 9. Easier than waiting for SAT? | **No** — SAT at least documents RGB whole-tooth **trained** weights (NC). This repo’s public `.pt` is COCO. |
| 10. Proceed? | **REJECTED** |

**Status: REJECTED**

Keep pursuing SegmentAnyTooth (NC email) and/or manual CVAT. Do not run 420-image inference with this `yolo11n.pt`.

---

## Safety

ICDAS modified = **NO**  
FDI generated = **NO**  
Tooth GT boxes = **0**  
Models trained = **NO**  
Original images modified = **NO**  
Lesion XML modified = **NO**  
420-image dataset modified = **NO**

**STOP.**
