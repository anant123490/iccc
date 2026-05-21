# Project Timeline and Implementation Report: ICDAS Dental Caries Detection

This report documents the alignment of the **ICDAS Dental Caries Detection** project implementation with the planned timeline. It outlines the specific features, architectural modules, and technical accomplishments mapped across **Phase 1** and **Phase 2**.

---

## 📊 High-Level Timeline & Status Summary

| Phase & Timeline | Planned Objectives | Codebase Status | Key Modules & Files |
| :--- | :--- | :--- | :--- |
| **Phase 1 (Weeks 1–4)** | Literature review, dataset collection, ICDAS annotation, preprocessing & augmentation pipeline | **100% Completed** | [preprocessing.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/preprocessing.py)<br>[augmentation.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/augmentation.py)<br>[setup_dataset.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/scripts/setup_dataset.py) |
| **Phase 1 (Weeks 5–8)** | MobileNetV3 training, spatial & channel attention integration, Grad-CAM visual testing | **100% Completed** | [model.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/model.py)<br>[attention.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/attention.py)<br>[gradcam.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/gradcam.py)<br>[train.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/train.py) |
| **Phase 1 (Weeks 9–12)** | Model evaluation on test set, SRS documentation, prototype PWA frontend development | **100% Completed** | [test_evaluation/](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/models/icdas_mobilenet_cbam/test_evaluation)<br>[frontend/](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend)<br>[docs/](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/docs) |
| **Phase 2 (Weeks 1–4)** | PWA service worker offline cache config, TensorFlow.js edge model conversion | **100% Completed** | [vite.config.ts](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/vite.config.ts)<br>[public/models/](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/public/models) |
| **Phase 2 (Weeks 5–8)** | End-to-end integration testing on devices, UI/UX refinement, secure local storage | **100% Completed** | [inference.ts](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/src/services/inference.ts)<br>[storage.ts](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/src/services/storage.ts)<br>[pages/](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/src/pages) |
| **Phase 2 (Weeks 9–12)** | Performance benchmarking, final documentation, project reporting | **100% Completed** | [export_report.json](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/models/export_report.json)<br>[PROJECT_REPORT.md](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/docs/PROJECT_REPORT.md) |

---

## 🔍 In-Depth Milestone Breakdown

### 📂 Phase 1 (Weeks 1–4): Data Engineering & Preprocessing Pipeline
*Focus: Creating a robust data ingestion, correction, and augmentation pipeline for high-fidelity clinical training.*

* **ICDAS Annotation & Manifest:** Structured [whatsapp_manifest.json](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/dataset/whatsapp_manifest.json) maps clinical images to ICDAS classes (0–6 ordinal scale) along with validation/test/train splits.
* **Pre-processing Suite ([preprocessing.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/preprocessing.py)):**
  * **Specular Reflection Reduction:** Attenuates camera flash/moisture glares from wet enamel to avoid feature distortion.
  * **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances boundary definitions between demineralized white spots (ICDAS 1/2) and sound enamel.
  * **ROI Cropping:** Automated cropping to extract standard oral areas.
* **Augmentation Suite ([augmentation.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/augmentation.py)):** Implements clinical-safe transforms (affine translations, rotations, random contrast/brightness scaling, elastic deformations) to prevent overfitting on specific camera lens shapes.

---

### 🧠 Phase 1 (Weeks 5–8): Deep Learning & Explainability Pipeline
*Focus: Modifying backbone models for attention-guided feature extraction and clinical interpretability.*

* **CBAM Attention ([attention.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/attention.py)):**
  * **Channel Attention:** Focuses on *what* spectral features (color shifts, spot contrast) indicate demineralization.
  * **Spatial Attention:** Focuses on *where* the lesion is located physically on the tooth.
* **Network & Loss Head ([model.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/model.py)):** Integrates MobileNetV3-Small backbone with a custom **Ordinal Classification Head** (addressing the progressive severity of caries rather than generic categorical classes) using Focal Loss for class imbalance.
* **Grad-CAM Integration ([gradcam.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/src/gradcam.py)):** Extracts activations from the final convolutional layer of MobileNetV3, computing the gradient score relative to the predicted ICDAS category. Generates custom heatmap layers to overlays on raw images in real time.

---

### 📈 Phase 1 (Weeks 9–12): Evaluation & Prototype PWA Frontend
*Focus: Validation of generalizability, documentation, and interface blueprinting.*

* **Rigorous Evaluation Reports ([test_evaluation/](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/models/icdas_mobilenet_cbam/test_evaluation)):**
  * **Confusion Matrix:** Shows high diagonal density indicating strong ordinal classification stability.
  * **ROC Curves:** Plotted and calculated per ICDAS class to demonstrate structural performance.
* **SRS and Architectural Integrity:** Created comprehensive docs for [ARCHITECTURE.md](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/docs/ARCHITECTURE.md), [TRAINING.md](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/docs/TRAINING.md), and [SETUP.md](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/docs/SETUP.md).
* **React + Vite + TypeScript PWA Prototype:** Complete folder structure in `frontend/` leveraging Outfit typography, dark-mode styling, and fully interactive layouts.

---

### 📶 Phase 2 (Weeks 1–4): Edge Performance & Offline Engine
*Focus: Converting models for resource-constrained browsers and configuring service workers.*

* **TensorFlow.js Converter ([export.py](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/ml/export.py)):** Converts the PyTorch-style Keras model into web-optimized Graph Models (`model.json` + binary chunks). Quantizes weights to secure a model size of `<20MB` for instant loading over mobile networks.
* **Service Worker Configuration ([vite.config.ts](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/vite.config.ts)):**
  * Implements `vite-plugin-pwa` in `autoUpdate` mode.
  * Explicitly caches static web content and heavy model files (`*.json`, `*.bin`, `*.wasm`) under a persistent Cache Storage system to ensure execution in isolated rural environments with **no internet access**.

---

### 🛡️ Phase 2 (Weeks 5–8): End-to-End Clinical Safety & Encryption
*Focus: Device testing, patient data privacy, and real-time frontend scanning logic.*

* **Edge Inference Engine ([inference.ts](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/src/services/inference.ts)):** Integrates TF.js with HTML5 Canvas API to perform instantaneous on-device image preprocessing, forward pass, and Grad-CAM color mapping.
* **AES-GCM Patient Privacy ([storage.ts](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/src/services/storage.ts)):**
  * Implements a local IndexedDB-backed history manager.
  * Uses Web Crypto API **AES-GCM (256-bit)** to encrypt all local patient reports, consent files, and annotations.
  * Guarantees 100% clinical data sovereignty on the doctor’s device (Zero Cloud dependency, aligns with HIPAA principles).
* **Rich Aesthetic UI Views:**
  * **[Home](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/src/pages/Home.tsx):** High-end clinical dashboard.
  * **[Scan](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/src/pages/Scan.tsx):** Beautiful interactive camera capture with real-time UI states.
  * **[Results](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/frontend/src/pages/Results.tsx):** Dynamic detail breakdown of caries grade, clinical action advice, and interactive explainability heatmap switches.

---

### ⚡ Phase 2 (Weeks 9–12): Performance Benchmarking & Reporting
*Focus: Measuring limits and finalizing formal research and reporting systems.*

* **Model Export Benchmarking ([export_report.json](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/models/export_report.json)):** Measures exact computation times, floating-point optimization benefits, and inference latency under different environments (Mean: `145.3ms`, p95: `416.4ms`).
* **Research and Reporting Drafts ([RESEARCH_PAPER_DRAFT.md](file:///c:/Users/anant/OneDrive/Desktop/icdas%20project/docs/RESEARCH_PAPER_DRAFT.md)):** Detailed drafting of methodology, model design (MobileNetV3 + CBAM attention + ordinal classification), results, and ethical design parameters.
