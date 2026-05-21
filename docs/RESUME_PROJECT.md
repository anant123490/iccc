# Resume-Ready Project Description

## One-Liner
Built an offline Progressive Web App for AI-powered dental caries detection using ICDAS ordinal classification with Grad-CAM explainability on edge devices.

## Bullet Points (Resume)

- Architected end-to-end **offline PWA** for ICDAS 0–6 dental caries classification from smartphone intraoral photos using **React, TypeScript, TensorFlow.js**, and **Service Workers**
- Designed **MobileNetV3-Small + CBAM attention** with **ordinal regression**, stratified K-fold training, focal loss, achieving **[X]%** accuracy and **[Y]** quadratic weighted kappa
- Implemented full **ML pipeline**: ROI detection, CLAHE preprocessing, specular reduction, augmentation, mixed-precision training, and **TFLite/TF.js** export with **<20MB** quantized models
- Built **Grad-CAM explainability** with heatmap overlays and lesion contour extraction for clinical interpretability
- Ensured **HIPAA-aligned privacy**: local-only storage, **AES-GCM encryption**, IndexedDB patient history, consent workflow — zero cloud dependency
- Deployed via **Docker**, **FastAPI** backend, **GitHub Actions CI/CD**; documented architecture, training, and deployment guides

## Skills Demonstrated
`TensorFlow` `PyTorch-style TF/Keras` `React` `TypeScript` `PWA` `FastAPI` `Computer Vision` `Medical AI` `Edge AI` `Grad-CAM` `Docker` `MLOps`

## GitHub README Summary (Short)
> Offline AI dental caries scanner — ICDAS classification from phone photos with explainable heatmaps. No internet required. Built with TF.js + MobileNetV3 + CBAM.
