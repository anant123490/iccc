# Demo Presentation Slides

---

## Slide 1: Title
# Offline AI Dental Caries Detection
### ICDAS Classification from Smartphone Photos
*Edge AI • Explainable • Privacy-First*

---

## Slide 2: Problem
- Dental caries: most common chronic disease
- ICDAS provides standardized severity scale (0–6)
- Need: accessible screening without specialist + internet

---

## Slide 3: Solution
| Feature | Benefit |
|---------|---------|
| Offline PWA | Works anywhere |
| <1s inference | Real-time feedback |
| Grad-CAM | Trust & interpretability |
| Local storage | Patient privacy |

---

## Slide 4: Architecture
```
Phone Camera → Preprocessing → MobileNetV3+CBAM → ICDAS Grade
                    ↓
              Grad-CAM Overlay → Clinical Action
```

---

## Slide 5: Live Demo
1. Open PWA on phone
2. Capture intraoral photo
3. View ICDAS grade + confidence
4. Show heatmap overlay
5. Save to patient history

---

## Slide 6: Model
- **Backbone:** MobileNetV3-Small
- **Attention:** CBAM
- **Head:** Ordinal regression
- **Size:** <20MB quantized

---

## Slide 7: Results
*[Insert your metrics after training]*
- Accuracy: __%
- Quadratic Kappa: __
- Latency: __ms

---

## Slide 8: Privacy & Compliance
- All data local
- AES encryption
- Consent screen
- **Disclaimer:** Clinical decision support only

---

## Slide 9: Tech Stack
React • TypeScript • TensorFlow.js • FastAPI • Docker • CI/CD

---

## Slide 10: Q&A
**GitHub:** [your-repo]
**Contact:** [your-email]

> This tool is for clinical decision support and is not a substitute for professional diagnosis.
