# Stage 3C — Roboflow 10-image feasibility test

Date: 2026-08-26  
Label: **CANDIDATE TEST OUTPUT — NOT GROUND TRUTH**  
Inference on the 10 images: **not run** (`ACCESS_REQUIRED` for both candidates).

Selected images path (verified):  
`c:\Users\anant\OneDrive\Desktop\icdas project\fdi_detection_dataset\images\selected\`  
**420** JPG files. Manifest: `annotation_project/manifests/selected_images.csv` with views  
`Maxillary_Occlusal`, `Mandibular`, `Frontal`, `Left_Lateral`, `Right_Lateral` (84 each).

Protected trees (`dataset/`, `ml/`, `models/`, app, Stage 3A/3B annotation files) were **not** modified. No YOLO/COCO/VOC overwrite. No weights/datasets downloaded. No packages installed.

---

## Environment (no installs)

| Package | Status |
| --- | --- |
| ultralytics | 8.4.45 |
| torch | 2.2.2+cpu |
| torchvision | import failed this check |
| opencv (cv2) | 4.11.0 |
| requests | 2.34.2 |
| roboflow | **MISSING — DO NOT INSTALL DURING THIS AUDIT** |
| inference_sdk | **MISSING — DO NOT INSTALL DURING THIS AUDIT** |
| `ROBOFLOW_API_KEY` | not used (not present in env) |

---

## Comparison table

| Criterion | Candidate A | Candidate B |
| --- | --- | --- |
| RGB intraoral | LIKELY (title/workspace); samples not inspected | LIKELY (title + 1213-image intraoral listing); samples not inspected |
| Whole-tooth detection | UNKNOWN (classes unverified) | UNKNOWN (classes unverified); type is instance segmentation |
| Model type | UNVERIFIED (page blocked) | Roboflow 3.0 Instance Segmentation (Fast) |
| Classes | UNVERIFIED | UNVERIFIED |
| FDI classes | UNVERIFIED; do not confuse with sibling FDI numbering project | UNVERIFIED |
| License | **LICENSE_STATUS = UNKNOWN** | **LICENSE_STATUS = UNKNOWN** |
| Weights accessible | No public `.pt` found without account | No public `.pt` found without account |
| Local inference | Not this audit | Not this audit |
| API required | Yes (documented pattern) | Yes (`API_KEY` in official snippet) |
| API key required | Yes | Yes |
| Tested on 10 images | **NO** | **NO** |
| Average predictions/image | n/a | n/a |
| Zero-prediction images | n/a | n/a |
| Obvious false positives | **Not judged** (no overlays) | **Not judged** |
| Technical readiness | Poor / Not testable here | Poor / Not testable here |
| Overall suitability | **Not testable** | **Not testable** |

Qualitative 10-image score: **Not testable** for A and B. No mAP. No accuracy claim.

---

## Advantages

- Both listings are **named** for intraoral tooth detection, unlike panoramic/X-ray pathology models.
- B documents **instance segmentation**, 1213 images, 640 stretch, model id `intraoral-tooth-detection-rohlq/1`.
- A sits on a workspace that also publishes intraoral numbering projects (still a **different** slug).

## Limitations

- Live Universe HTML was **Cloudflare-blocked**; class names and licenses were **not** verified on the actual pages.
- Hosted inference needs **API key** + `inference-sdk` (missing; not installed).
- No 10-image visualizations.

## License risks

**UNKNOWN** for both. Sibling Roboflow projects sometimes show CC BY 4.0 — **not copied onto these two slugs**. Do not claim legal usability or commercial deployment.

## Technical risks

Wrong classes (FDI vs tooth vs pathology); 640 stretch; API/cloud; AGPL if later exported to YOLO; domain shift vs Zenodo photos.

## Project compatibility

Until a **human** logs into Universe, confirms **license + class list**, and a **later authorized** test uses a key, **neither candidate can fill Stage 3C boxes**. Existing empty YOLO/COCO/VOC files stay empty.

---

## 10-image sample (paths only, originals not copied)

See `test_images.csv`. Two files per view, first two filenames per `mouth_view` in the Stage 3B manifest.

Predictions: `prediction_summary.csv` — all rows `ACCESS_REQUIRED`.

---

## Final questions

1. **Technically more suitable?** **B is better documented** (seg type, image count, model id). **Neither was executed.**  
2. **Clearest usable license?** **Neither** — both **UNKNOWN**.  
3. **Testable without restricted access?** **Neither.**  
4. **Better-looking detections on the 10 images?** **Cannot say** — no inference.  
5. **Suitable to generate candidate boxes for human QC?** **Not on current evidence.** Need license + classes + authorized test first.  
6. **Is SegmentAnyTooth still worth pursuing?** **YES** — still the only audited RGB whole-tooth detector with a documented (NC, email-gated) weight path.  
7. **Is manual annotation still necessary?** **YES** (CVAT/QC). Pretrained outputs would still need humans.  
8. **Test all 420 images now?** **NO.**

Do **not** auto-start Option 1–3. Next step is **human review** of Universe pages (license/classes) and/or SAT weight request.

---

## Safety check

| Item | Result |
| --- | --- |
| ICDAS modified | NO |
| FDI labels generated | 0 |
| Tooth GT boxes generated | 0 |
| Models trained | NO |
| Original images modified | NO |
| Lesion XML modified | NO |
| Stage 3A/3B annotation files modified | NO |
| External datasets downloaded | NO |
| External model weights downloaded | NO |

**STOP.**
