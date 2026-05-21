# Offline AI-Based Dental Caries Detection using ICDAS Classification

## Abstract

Dental caries remains the most prevalent chronic disease worldwide. The International Caries Detection and Assessment System (ICDAS) provides a standardized ordinal scale for lesion severity, but clinical application requires trained examiners. We present an offline-capable Progressive Web App leveraging Edge AI — MobileNetV3-Small with Convolutional Block Attention Module (CBAM) and ordinal regression — to classify intraoral smartphone photographs into ICDAS grades 0–6. Our system achieves explainable predictions via Grad-CAM heatmaps with lesion contour extraction, stores all data locally with AES encryption, and targets sub-second inference on low-end Android devices through TensorFlow Lite and TensorFlow.js quantization.

## 1. Introduction

Early caries detection reduces restorative treatment burden. Smartphone intraoral photography enables screening in resource-limited settings, but cloud-dependent AI solutions raise privacy concerns and fail offline. We address these limitations with a fully offline PWA architecture.

## 2. Related Work

Prior work on dental caries detection predominantly uses binary classification on radiographs or extraoral images. ICDAS ordinal labeling has received limited attention in deep learning literature due to dataset scarcity.

## 3. Methodology

### 3.1 Preprocessing
Tooth ROI detection, CLAHE contrast enhancement, specular reflection inpainting, and color normalization simulate clinical imaging consistency.

### 3.2 Model
MobileNetV3-Small backbone with CBAM attention. Ordinal regression via cumulative link model preserves label ordering. Focal loss addresses class imbalance.

### 3.3 Training
Stratified 5-fold cross-validation, mixed precision, cosine learning rate schedule, early stopping.

### 3.4 Explainability
Grad-CAM visualizes discriminative regions; morphological contour extraction highlights suspected lesions.

### 3.5 Edge Deployment
Dynamic range quantization reduces model size below 20MB. Benchmarked latency on mobile CPUs.

## 4. Experiments

*[Fill after training on your dataset]*

| Metric | Value |
|--------|-------|
| Accuracy | — |
| Weighted F1 | — |
| Quadratic Kappa | — |
| Inference (p95) | — ms |

## 5. Conclusion

We demonstrate a privacy-preserving, offline-first clinical decision support tool for ICDAS classification. Future work includes multi-tooth detection and prospective clinical validation.

## References

1. Pitts NB, et al. ICDAS II consensus. Community Dent Oral Epidemiol. 2007.
2. Howard A, et al. Searching for MobileNetV3. ICCV 2019.
3. Woo S, et al. CBAM. ECCV 2018.
