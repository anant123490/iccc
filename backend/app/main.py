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


# =========================================================
# SETTINGS
# =========================================================

settings = get_settings()


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="ICDAS Dental Caries Detection API",
    description=(
        "Offline-capable AI inference for ICDAS classification "
        "with Grad-CAM explainability and AI-assisted report generation."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# INFERENCE ENGINE
# =========================================================

engine: InferenceEngine | None = None


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup():

    global engine

    model_path = settings.deploy_model_path

    import os

    # Prefer deploy model
    if not os.path.exists(model_path):
        model_path = settings.model_path

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model file not found. "
            f"Checked: {settings.deploy_model_path} "
            f"and {settings.model_path}"
        )

    engine = InferenceEngine.get_instance(
        model_path=model_path,
        num_classes=settings.num_classes,
        image_size=settings.image_size,
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {
        "message": "ICDAS Dental Caries Detection API",
        "status": "running",
        "version": "1.0.0",
    }


# =========================================================
# HEALTH
# =========================================================

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


# =========================================================
# MODEL INFORMATION
# =========================================================

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


# =========================================================
# ICDAS PREDICTION
# =========================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    include_explainability: bool = True,
):
    """
    Upload an intraoral image for ICDAS prediction.

    Returns:

    - ICDAS grade
    - confidence
    - clinical action
    - finding
    - recommendation
    - urgency
    - probabilities
    - optional Grad-CAM results
    """

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if engine is None:

        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )


    # -----------------------------------------------------
    # Read uploaded file
    # -----------------------------------------------------

    content = await file.read()


    # -----------------------------------------------------
    # Validate file size
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Validate empty file
    # -----------------------------------------------------

    if len(content) == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    try:

        # Preprocess uploaded image
        original, processed = (
            engine.preprocess_upload(content)
        )


        # Run model prediction
        result = engine.predict(processed)


        # -------------------------------------------------
        # Get ICDAS clinical information
        # -------------------------------------------------

        action = get_clinical_action(
            result["icdas_grade"]
        )


        # -------------------------------------------------
        # Build response
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Grad-CAM
        # -------------------------------------------------

        if include_explainability:

            explain = engine.explain(
                processed,
                original,
                result["icdas_grade"],
            )

            response.heatmap_base64 = (
                explain["heatmap"]
            )

            response.overlay_base64 = (
                explain["overlay"]
            )

            response.contour_base64 = (
                explain["contour"]
            )


        return response


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}",
        )


# =========================================================
# AI REPORT GENERATION
# =========================================================

@app.post(
    "/api/v1/report",
    response_model=ReportResponse,
)
async def create_report(
    data: ReportRequest,
):
    """
    Generate structured AI-assisted dental information.

    Important:

    Groq returns structured information only.

    The frontend is responsible for creating
    the visual AI suggestion card.
    """

    try:

        # -------------------------------------------------
        # Ask Groq for structured information
        # -------------------------------------------------

        ai_result = generate_report(

            icdas_grade=data.icdas_grade,

            confidence=data.confidence,
        )


        # -------------------------------------------------
        # Return structured response
        # -------------------------------------------------

        return ReportResponse(

            icdas_grade=data.icdas_grade,

            confidence=data.confidence,

            finding=ai_result["finding"],

            recommendation=(
                ai_result["recommendation"]
            ),

            urgency=ai_result["urgency"],
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Report generation failed: {str(e)}"
            ),
        )


        "anant"