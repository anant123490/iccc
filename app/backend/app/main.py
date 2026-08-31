"""
FastAPI backend for ICDAS dental caries detection.

Supported ICDAS range:
    0-4

Features:
    - Image upload
    - 5-class model inference
    - Confidence calculation
    - Low-confidence detection
    - Grad-CAM explainability
    - Groq report generation
    - Prediction history
    - Statistics
    - SQLite/PostgreSQL database support
"""

from __future__ import annotations

import cv2
import numpy as np
import base64
import json
import logging
import os
import uuid
from collections import Counter
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import get_settings
from .database import SessionLocal, init_db
from .db_models import PredictionRecord
from .caries_pipeline import run_localized_pipeline
from .groq_service import (
    generate_report,
    groq_configured,
)
from .icdas_actions import get_clinical_action
from .inference import InferenceEngine
from .schemas import (
    AnalyzeResponse,
    HealthResponse,
    HistoryDetail,
    HistoryItem,
    ModelInfoResponse,
    PredictionResponse,
    ReportRequest,
    ReportResponse,
    StatsResponse,
)
from .portal_routes import router as portal_router
from .storage_paths import ensure_dirs


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s: "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "icdas.api"
)


# ============================================================
# SETTINGS
# ============================================================

settings = get_settings()


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = (
    Path(__file__).resolve().parents[1]
    / "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ICDAS Dental Caries Detection API",
    description=(
        "AI inference for ICDAS 0-4 classification "
        "with Grad-CAM explainability and "
        "Groq-generated decision-support reports."
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portal_router)


# ============================================================
# GLOBAL INFERENCE ENGINE
# ============================================================

engine: InferenceEngine | None = None


# ============================================================
# UTILITY: ENCODE FILE
# ============================================================

def _encode_file(
    path: str | None,
) -> str | None:

    if not path:
        return None

    file_path = Path(path)

    if not file_path.exists():
        return None

    try:

        return base64.b64encode(
            file_path.read_bytes()
        ).decode(
            "utf-8"
        )

    except Exception:

        logger.exception(
            "Could not encode file: %s",
            path,
        )

        return None


# ============================================================
# UTILITY: SAVE BYTES
# ============================================================

def _save_bytes(
    data: bytes,
    suffix: str,
) -> str:

    name = (
        f"{uuid.uuid4().hex}"
        f"{suffix}"
    )

    destination = (
        UPLOAD_DIR / name
    )

    destination.write_bytes(
        data
    )

    return str(
        destination
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup():

    global engine

    logger.info(
        "Starting ICDAS API..."
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    try:

        init_db()
        ensure_dirs()

        logger.info(
            "Database initialized."
        )

    except Exception as exc:

        logger.exception(
            "Database initialization failed."
        )

        raise exc

    # --------------------------------------------------------
    # Determine model
    # --------------------------------------------------------

    deploy_model = Path(
        settings.deploy_model_path
    )

    regular_model = Path(
        settings.model_path
    )

    logger.info(
        "Configured deploy model: %s",
        deploy_model,
    )

    logger.info(
        "Configured regular model: %s",
        regular_model,
    )

    engine = None

    # --------------------------------------------------------
    # Prefer deploy model
    # --------------------------------------------------------

    if deploy_model.exists():

        model_path = deploy_model

    elif regular_model.exists():

        model_path = regular_model

    else:

        model_path = None

        logger.warning(
            "No 5-class softmax deploy.keras at configured paths."
        )

        engine = None

    from .portal_runtime import is_blocked_icdas_checkpoint

    if (
        model_path is not None
        and not is_blocked_icdas_checkpoint(model_path)
    ):

        try:

            logger.info(
                "Loading trained ICDAS model: %s",
                model_path,
            )

            engine = InferenceEngine.get_instance(
                model_path=str(
                    model_path
                ),
                num_classes=settings.num_classes,
                image_size=settings.image_size,
                ordinal_regression=(
                    settings.ordinal_regression
                ),
                confidence_threshold=(
                    settings.confidence_threshold
                ),
            )

            logger.info(
                "ICDAS inference engine loaded successfully."
            )

        except Exception:

            logger.exception(
                "Could not initialize 5-class inference engine."
            )

            engine = None

    from .portal_runtime import load_portal_runtime

    runtime = load_portal_runtime()
    app.state.runtime = runtime
    if runtime.engine is not None:
        engine = runtime.engine
    app.state.engine = engine
    st = runtime.status()
    logger.info("Tooth Detector V2: %s", st.get("detector_status"))
    logger.info("ICDAS classifier: %s", st.get("icdas_status"))


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": (
            "ICDAS Dental Caries Detection API"
        ),
        "status": (
            "running"
        ),
        "version": (
            "2.0.0"
        ),
        "icdas_mode": (
            "0-4"
        ),
        "num_classes": (
            settings.num_classes
        ),
        "model_loaded": (
            engine is not None
            and engine.model is not None
        ),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
)
@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health():

    db_ok = True

    db = None

    try:

        db = SessionLocal()

        db.execute(
            text("SELECT 1")
        )

    except Exception:

        logger.exception(
            "Database health check failed."
        )

        db_ok = False

    finally:

        if db is not None:
            db.close()

    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        from .portal_runtime import get_runtime

        runtime = get_runtime()
    st = runtime.status()
    icdas_loaded = bool(st.get("icdas_loaded"))
    detector_ok = bool(st.get("detector_v2"))

    return HealthResponse(
        status=(
            "healthy"
            if detector_ok and db_ok
            else "degraded"
        ),
        model_loaded=icdas_loaded,
        database_ok=db_ok,
        groq_configured=(
            groq_configured()
        ),
        tooth_detector_v2=str(st.get("detector_status") or "UNAVAILABLE"),
        icdas_classifier=str(
            st.get("icdas_status") or "NOT_TRAINED / NOT_DEPLOYED"
        ),
    )


# ============================================================
# MODEL INFO
# ============================================================

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
        ordinal_regression=(
            settings.ordinal_regression
        ),
    )


# ============================================================
# INTERNAL PREDICTION FUNCTION
# ============================================================

def _run_prediction(
    content: bytes,
    filename: str,
    include_explainability: bool,
    user_id: str,
) -> PredictionResponse:

    # --------------------------------------------------------
    # Engine
    # --------------------------------------------------------

    if (
        engine is None
        or engine.model is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Inference engine is unavailable. "
                "Make sure a trained 5-class ICDAS model "
                "is configured."
            ),
        )

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    max_bytes = (
        settings.max_upload_mb
        * 1024
        * 1024
    )

    if len(content) > max_bytes:

        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds "
                f"{settings.max_upload_mb}MB limit."
            ),
        )

    # --------------------------------------------------------
    # Empty file
    # --------------------------------------------------------

    if len(content) == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # File type
    # --------------------------------------------------------

    suffix = (
        Path(
            filename or "upload.jpg"
        )
        .suffix
        .lower()
    )

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    if suffix not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Please upload JPG, PNG, BMP, "
                "or WEBP."
            ),
        )

    logger.info(
        "Prediction request received: "
        "size=%s bytes user=%s filename=%s",
        len(content),
        user_id,
        filename,
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    try:

        original, processed = (
            engine.preprocess_upload(
                content
            )
        )

    except ValueError as exc:

        logger.warning(
            "Image preprocessing failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid or corrupted image: {exc}"
            ),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected preprocessing failure."
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to process uploaded image."
            ),
        ) from exc

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        result = engine.predict(
            processed
        )

    except Exception as exc:

        logger.exception(
            "Model prediction failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Model unavailable or prediction failed."
            ),
        ) from exc

    # --------------------------------------------------------
    # Clinical action
    # --------------------------------------------------------

    try:

        action = get_clinical_action(
            result["icdas_grade"]
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Groq report
    # --------------------------------------------------------

    report_data = None

    try:

        report_data = generate_report(
            icdas_grade=result[
                "icdas_grade"
            ],
            confidence=result[
                "confidence"
            ],
            finding=action[
                "finding"
            ],
            urgency=action[
                "urgency"
            ],
        )

    except Exception:

        logger.exception(
            "Groq report generation failed."
        )

        report_data = None

    # --------------------------------------------------------
    # Explainability
    # --------------------------------------------------------

    heatmap_b64 = None
    overlay_b64 = None
    contour_b64 = None
    heatmap_path = None

    if include_explainability:

        try:

            explain = engine.explain(
                processed=processed,
                original_rgb=original,
                predicted_grade=result[
                    "icdas_grade"
                ],
            )

            heatmap_b64 = (
                explain["heatmap"]
            )

            overlay_b64 = (
                explain["overlay"]
            )

            contour_b64 = (
                explain["contour"]
            )

            # Save heatmap.
            heatmap_path = _save_bytes(
                base64.b64decode(
                    heatmap_b64
                ),
                ".png",
            )

        except Exception:

            logger.exception(
                "Grad-CAM generation failed."
            )

            # Prediction should still succeed
            # even if explainability fails.

    # --------------------------------------------------------
    # Save original image
    # --------------------------------------------------------

    image_path = _save_bytes(
        content,
        suffix or ".jpg",
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    record_id = None
    db: Session | None = None

    try:

        db = SessionLocal()

        record = PredictionRecord(
            user_id=(
                user_id
                or "anonymous"
            ),
            image_path=image_path,
            heatmap_path=heatmap_path,
            icdas_grade=result[
                "icdas_grade"
            ],
            confidence=result[
                "confidence"
            ],
            urgency=(
                report_data or {}
            ).get(
                "urgency",
                action[
                    "urgency"
                ],
            ),
            finding=(
                report_data or {}
            ).get(
                "finding",
                action[
                    "finding"
                ],
            ),
            recommendation=(
                report_data or {}
            ).get(
                "recommendation",
                action[
                    "recommendation"
                ],
            ),
            report=(
                report_data or {}
            ).get(
                "report"
            ),
            probabilities_json=json.dumps(
                result[
                    "probabilities"
                ]
            ),
        )

        db.add(
            record
        )

        db.commit()

        db.refresh(
            record
        )

        record_id = record.id

    except Exception:

        logger.exception(
            "Database failure while "
            "storing prediction."
        )

        if db is not None:
            db.rollback()

    finally:

        if db is not None:
            db.close()

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        "Prediction complete: "
        "grade=%s confidence=%.2f id=%s",
        result[
            "icdas_grade"
        ],
        result[
            "confidence"
        ],
        record_id,
    )

    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return PredictionResponse(
        id=record_id,
        icdas_grade=result[
            "icdas_grade"
        ],
        confidence=result[
            "confidence"
        ],
        label=action[
            "label"
        ],
        action=action[
            "action"
        ],
        description=action[
            "description"
        ],
        finding=(
            report_data or {}
        ).get(
            "finding",
            action[
                "finding"
            ],
        ),
        recommendation=(
            report_data or {}
        ).get(
            "recommendation",
            action[
                "recommendation"
            ],
        ),
        urgency=(
            report_data or {}
        ).get(
            "urgency",
            action[
                "urgency"
            ],
        ),
        probabilities=result[
            "probabilities"
        ],
        low_confidence=result[
            "low_confidence"
        ],
        low_confidence_message=result[
            "low_confidence_message"
        ],
        report=(
            report_data or {}
        ).get(
            "report"
        ),
        heatmap_base64=heatmap_b64,
        overlay_base64=overlay_b64,
        contour_base64=contour_b64,
    )


# ============================================================
# PREDICT
# ============================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
)
@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    include_explainability: bool = True,
    x_user_id: str | None = Header(
        default="anonymous"
    ),
):

    content = await file.read()

    return _run_prediction(
        content=content,
        filename=(
            file.filename
            or "upload.jpg"
        ),
        include_explainability=(
            include_explainability
        ),
        user_id=(
            x_user_id
            or "anonymous"
        ),
    )


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
)
async def analyze_localized(
    file: UploadFile = File(...),
    include_explainability: bool = True,
    allow_whole_image_fallback: bool = False,
    x_user_id: str | None = Header(
        default="anonymous"
    ),
):
    """Decay-region detector then ICDAS per ROI. Does not map d/D to ICDAS."""

    if engine is None or engine.model is None:
        raise HTTPException(
            status_code=503,
            detail="ICDAS inference engine is unavailable.",
        )

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large.")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        bgr = engine._decode_image_bytes(content)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = run_localized_pipeline(
        engine=engine,
        image_rgb=rgb,
        include_explainability=include_explainability,
        allow_whole_image_fallback=allow_whole_image_fallback,
    )
    payload["id"] = None
    return AnalyzeResponse.model_validate(payload)


# ============================================================
# REPORT
# ============================================================

@app.post(
    "/api/v1/report",
    response_model=ReportResponse,
)
async def create_report(
    data: ReportRequest,
):

    try:

        ai_result = generate_report(
            icdas_grade=data.icdas_grade,
            confidence=data.confidence,
            finding=data.finding,
            urgency=data.urgency,
            model_name=(
                data.model_name
                or (
                    "MobileNetV3-Small + "
                    "CBAM + 5-class softmax"
                )
            ),
        )

        return ReportResponse(
            icdas_grade=data.icdas_grade,
            confidence=data.confidence,
            finding=ai_result[
                "finding"
            ],
            recommendation=ai_result[
                "recommendation"
            ],
            urgency=ai_result[
                "urgency"
            ],
            report=ai_result[
                "report"
            ],
        )

    except Exception as exc:

        logger.exception(
            "Report generation failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Report generation failed.",
        ) from exc


# ============================================================
# HISTORY
# ============================================================

@app.get(
    "/api/v1/history",
    response_model=list[HistoryItem],
)
async def list_history(
    limit: int = Query(
        50,
        ge=1,
        le=200,
    ),
):

    db = SessionLocal()

    try:

        rows = (
            db.query(
                PredictionRecord
            )
            .order_by(
                PredictionRecord.created_at.desc()
            )
            .limit(
                limit
            )
            .all()
        )

        return [
            HistoryItem(
                id=row.id,
                created_at=row.created_at,
                icdas_grade=row.icdas_grade,
                confidence=row.confidence,
                urgency=row.urgency,
                finding=row.finding,
                has_image=bool(
                    row.image_path
                ),
                has_heatmap=bool(
                    row.heatmap_path
                ),
            )
            for row in rows
        ]

    except Exception as exc:

        logger.exception(
            "Database failure while "
            "listing history."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Could not load "
                "prediction history."
            ),
        ) from exc

    finally:

        db.close()


# ============================================================
# HISTORY DETAIL
# ============================================================

@app.get(
    "/api/v1/history/{prediction_id}",
    response_model=HistoryDetail,
)
async def get_history_item(
    prediction_id: int,
):

    db = SessionLocal()

    try:

        row = db.get(
            PredictionRecord,
            prediction_id,
        )

        if row is None:

            raise HTTPException(
                status_code=404,
                detail="Prediction not found.",
            )

        probs = (
            json.loads(
                row.probabilities_json
            )
            if row.probabilities_json
            else None
        )

        return HistoryDetail(
            id=row.id,
            created_at=row.created_at,
            icdas_grade=row.icdas_grade,
            confidence=row.confidence,
            urgency=row.urgency,
            finding=row.finding,
            recommendation=row.recommendation,
            report=row.report,
            probabilities=probs,
            has_image=bool(
                row.image_path
            ),
            has_heatmap=bool(
                row.heatmap_path
            ),
            image_base64=_encode_file(
                row.image_path
            ),
            heatmap_base64=_encode_file(
                row.heatmap_path
            ),
        )

    finally:

        db.close()


# ============================================================
# STATS
# ============================================================

@app.get(
    "/api/v1/stats",
    response_model=StatsResponse,
)
async def stats():

    db = SessionLocal()

    try:

        rows = (
            db.query(
                PredictionRecord
            )
            .all()
        )

        if not rows:

            return StatsResponse(
                total_analyses=0,
                average_confidence=0.0,
                most_common_grade=None,
                high_severity_cases=0,
                grade_distribution={
                    str(i): 0
                    for i in range(
                        settings.num_classes
                    )
                },
                confidence_buckets={
                    "0-50": 0,
                    "50-70": 0,
                    "70-85": 0,
                    "85-100": 0,
                },
            )

        grades = [
            row.icdas_grade
            for row in rows
        ]

        confs = [
            row.confidence
            for row in rows
        ]

        dist = Counter(
            grades
        )

        buckets = {
            "0-50": 0,
            "50-70": 0,
            "70-85": 0,
            "85-100": 0,
        }

        for confidence in confs:

            if confidence < 50:

                buckets[
                    "0-50"
                ] += 1

            elif confidence < 70:

                buckets[
                    "50-70"
                ] += 1

            elif confidence < 85:

                buckets[
                    "70-85"
                ] += 1

            else:

                buckets[
                    "85-100"
                ] += 1

        return StatsResponse(
            total_analyses=len(
                rows
            ),
            average_confidence=round(
                sum(confs)
                / len(confs),
                2,
            ),
            most_common_grade=max(
                dist,
                key=dist.get,
            ),
            high_severity_cases=sum(
                1
                for grade in grades
                if grade >= 3
            ),
            grade_distribution={
                str(i): int(
                    dist.get(
                        i,
                        0,
                    )
                )
                for i in range(
                    settings.num_classes
                )
            },
            confidence_buckets=buckets,
        )

    except Exception as exc:

        logger.exception(
            "Database failure while "
            "computing analytics."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Could not load analytics."
            ),
        ) from exc

    finally:

        db.close()