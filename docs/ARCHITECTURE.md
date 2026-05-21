# System Architecture

## High-Level Design

```mermaid
flowchart TB
    subgraph Client["PWA Client"]
        UI[React UI]
        CAM[Camera / Gallery]
        TFJS[TensorFlow.js]
        IDB[(IndexedDB)]
        SW[Service Worker]
    end

    subgraph Optional["Optional Backend"]
        API[FastAPI]
        Keras[Keras Model]
        GC[Grad-CAM]
    end

    subgraph Training["ML Pipeline"]
        Data[Dataset Loader]
        Pre[Preprocessing]
        Model[MobileNetV3 + CBAM]
        Export[TFLite / TF.js]
    end

    CAM --> UI
    UI --> TFJS
    TFJS --> IDB
    SW --> UI
    UI -.->|optional| API
    API --> Keras --> GC
    Data --> Pre --> Model --> Export
    Export --> TFJS
    Export --> Keras
```

## Model Architecture

```mermaid
flowchart LR
    Input[224x224 RGB] --> MB[MobileNetV3 Small]
    MB --> CBAM[CBAM Attention]
    CBAM --> GAP[Global Avg Pool]
    GAP --> FC[Dense 256]
    FC --> Ord[Ordinal Head K-1]
    FC --> Cls[Softmax K classes]
```

## Data Flow (Inference)

1. Capture/upload intraoral image
2. Preprocess: ROI crop → CLAHE → specular reduction → resize
3. TF.js inference (<1s target)
4. Grad-CAM heatmap + contour overlay
5. Encrypt & save to IndexedDB
6. Display ICDAS grade + clinical action

## Component Responsibilities

| Module | Responsibility |
|--------|----------------|
| `ml/src/preprocessing.py` | Image normalization |
| `ml/src/model.py` | Architecture definition |
| `ml/src/gradcam.py` | Explainability |
| `frontend/src/services/inference.ts` | Edge inference |
| `frontend/src/services/storage.ts` | Offline persistence |
| `backend/app/inference.py` | Server-side inference |

## Privacy Architecture

- Default: 100% on-device
- Encryption: Web Crypto AES-GCM
- No external network calls in offline mode
- Backend disabled unless user opts in
