# Stage 3C — External RGB tooth-detector test — FINAL DECISION

Date: 2026-08-26

**NO_RELIABLE_EXTERNAL_MODEL**

Did **not** run 10-image inference. Did **not** run 420 images. Did **not** write `annotations/candidates/`. Did **not** train. ICDAS/lesion XML/Stage 3A–3B labels untouched.

Selected set: **420** files in `fdi_detection_dataset/images/selected/`. Class for *our* dataset remains **`0 = tooth`**.

---

## Candidates (documentation only)

| # | Model | Type (if documented) | License | Access | Used for boxes? |
| --- | --- | --- | --- | --- | --- |
| 1 | [dental-cdueb/intraoral-tooth-detection-rohlq/1](https://universe.roboflow.com/dental-cdueb/intraoral-tooth-detection-rohlq) | Roboflow 3.0 instance segmentation (Fast); 1213 images; API_KEY in official snippet | **LICENSE_UNKNOWN** | **ACCESS_REQUIRED** | No |
| 2 | [dentalmate6v/intraoral-tooth-detection/1](https://universe.roboflow.com/dentalmate6v/intraoral-tooth-detection) | Workspace lists ~1.21k images / 1 model; classes unverified | **LICENSE_UNKNOWN** | **ACCESS_REQUIRED** | No |
| 3 | [tooth-detection-9ayo9/rgb-teeth-oj2v2](https://universe.roboflow.com/tooth-detection-9ayo9/rgb-teeth-oj2v2) | Name suggests RGB teeth; **page not verified** (Cloudflare) | **LICENSE_UNKNOWN** | **ACCESS_REQUIRED** | No |
| 4 | [ti-tvysg/rgb-teeth-iael6](https://universe.roboflow.com/ti-tvysg/rgb-teeth-iael6) | Name suggests RGB teeth; **page not verified** | **LICENSE_UNKNOWN** | **ACCESS_REQUIRED** | No |

Do not assume “tooth” in the slug means whole-tooth RGB detection. Sibling/other Universe projects were **not** substituted.

All four: hosted Roboflow inference typically needs `inference_sdk` + API key. Environment: `inference_sdk` **missing**, `ROBOFLOW_API_KEY` **unset**. Not bypassed. No weight downloads.

---

## Required answers

1. **Technically accessible?** **None** (no SDK, no key, no local public `.pt`).  
2. **Acceptable licensing?** **None verified.** LICENSE_UNKNOWN → not used.  
3. **Best on 10-image test?** **None tested.**  
4. **How many teeth detected?** **0** (no inference).  
5. **Suspicious predictions?** **0** (none generated).  
6. **Suitable for 420-image candidate generation?** **No.**  
7. **Does human work reduce to QC rather than drawing all boxes?** **Not yet.** Without candidates, Batch_01 still means **drawing** boxes in CVAT (`STAGE3C_MANUAL_ANNOTATION.md`).  
8. **Seed-annotation strategy:** Annotate **Batch_01 (60 images)** manually in CVAT, class `tooth` only. After a verified seed, a later stage can train **our** detector (Phase 10) — **not now**. SegmentAnyTooth remains a gated NC option, not used here.

Comparison table: `model_comparison.csv`. Test files: `test_images.csv`. Visualizations: none.

**STOP.** Do not proceed to 420-image inference or training until a human confirms license + access on a live Universe page.
