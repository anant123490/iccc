"""
FastAPI backend for ICDAS dental caries detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings

from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReportRequest,
    ReportResponse,
)

from .inference import InferenceEngine
from .icdas_actions import get_clinical_action
from .groq_service import generate_report


settings = get_settings()


app = FastAPI(
    title="ICDAS Dental Caries Detection API",
    description=(
        "Offline-capable AI inference for ICDAS classification "
        "with Grad-CAM explainability and AI-assisted report generation."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Inference engine
# ---------------------------------------------------------

engine: InferenceEngine | None = None


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
async def startup():
    global engine

    model_path = settings.deploy_model_path

    import os

    if not os.path.exists(model_path):
        model_path = settings.model_path

    engine = InferenceEngine.get_instance(
        model_path=model_path,
        num_classes=settings.num_classes,
        image_size=settings.image_size,
    )


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
)
async def health():

    return HealthResponse(
        status="healthy",
        model_loaded=(
            engine is not None
            and engine.model is not None
        ),
    )


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

@app.get(
    "/api/v1/model/info",
    response_model=ModelInfoResponse,
)
async def model_info():

    return ModelInfoResponse(
        name="icdas_mobilenet_cbam",
        num_classes=settings.num_classes,
        icdas_mode=settings.icdas_mode,
        image_size=settings.image_size,
    )


# ---------------------------------------------------------
# ICDAS prediction
# ---------------------------------------------------------

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    include_explainability: bool = True,
):
    """
    Upload intraoral image for ICDAS prediction.

    Returns:
    - ICDAS grade
    - confidence
    - clinical action
    - optional Grad-CAM overlays
    """

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )

    content = await file.read()

    max_bytes = (
        settings.max_upload_mb * 1024 * 1024
    )

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds "
                f"{settings.max_upload_mb}MB limit"
            ),
        )

    try:

        original, processed = engine.preprocess_upload(
            content
        )

        result = engine.predict(
            processed
        )

        action = get_clinical_action(
            result["icdas_grade"]
        )

        response = PredictionResponse(
            icdas_grade=result["icdas_grade"],
            confidence=result["confidence"],
            label=action["label"],
            action=action["action"],
            description=action["description"],
            finding=action["finding"],
            recommendation=action["recommendation"],
            urgency=action["urgency"],
            probabilities=result["probabilities"],
        )

        if include_explainability:

            explain = engine.explain(
                processed,
                original,
                result["icdas_grade"],
            )

            response.heatmap_base64 = explain["heatmap"]
            response.overlay_base64 = explain["overlay"]
            response.contour_base64 = explain["contour"]

        return response

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}",
        )


# ---------------------------------------------------------
# AI REPORT GENERATION
# ---------------------------------------------------------

@app.post(
    "/api/v1/report",
    response_model=ReportResponse,
)
async def create_report(
    data: ReportRequest,
):
    """
    Generate an AI-assisted dental report using Groq.

    The ICDAS grade and confidence are provided by the
    existing machine-learning prediction system.
    """

    try:

        report = generate_report(
            icdas_grade=data.icdas_grade,
            confidence=data.confidence,
        )

        return ReportResponse(
            report=report
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}",
        )