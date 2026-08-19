"""
FastAPI backend for ICDAS dental caries detection (ICDAS 0–4).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import get_settings
from .database import SessionLocal, init_db
from .db_models import PredictionRecord
from .groq_service import generate_report, groq_configured
from .icdas_actions import get_clinical_action
from .inference import InferenceEngine
from .schemas import (
    HealthResponse,
    HistoryDetail,
    HistoryItem,
    ModelInfoResponse,
    PredictionResponse,
    ReportRequest,
    ReportResponse,
    StatsResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("icdas.api")

settings = get_settings()
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="ICDAS Dental Caries Detection API",
    description=(
        "AI inference for ICDAS 0–4 classification with Grad-CAM "
        "explainability and Groq-generated decision-support reports."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: InferenceEngine | None = None


def _encode_file(path: str | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def _save_bytes(data: bytes, suffix: str) -> str:
    name = f"{uuid.uuid4().hex}{suffix}"
    dest = UPLOAD_DIR / name
    dest.write_bytes(data)
    return str(dest)


@app.on_event("startup")
async def startup():
    global engine
    init_db()

    model_path = settings.deploy_model_path
    if not os.path.exists(model_path):
        model_path = settings.model_path
    if not os.path.exists(model_path):
        logger.warning(
            "Model file not found at %s or %s; building an untrained 5-class model.",
            settings.deploy_model_path,
            settings.model_path,
        )
        engine = InferenceEngine.get_instance(
            model_path=model_path,
            num_classes=settings.num_classes,
            image_size=settings.image_size,
            ordinal_regression=settings.ordinal_regression,
            confidence_threshold=settings.confidence_threshold,
        )
        logger.info("Untrained 5-class model constructed for demo inference.")
        return

    logger.info("Loading model from %s", model_path)
    engine = InferenceEngine.get_instance(
        model_path=model_path,
        num_classes=settings.num_classes,
        image_size=settings.image_size,
        ordinal_regression=settings.ordinal_regression,
        confidence_threshold=settings.confidence_threshold,
    )
    logger.info("Model loading complete.")


@app.get("/")
async def root():
    return {
        "message": "ICDAS Dental Caries Detection API",
        "status": "running",
        "version": "2.0.0",
        "icdas_mode": "0-4",
        "num_classes": settings.num_classes,
    }


@app.get("/api/v1/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health():
    db_ok = True
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception:
        logger.exception("Database health check failed")
        db_ok = False
    return HealthResponse(
        status="healthy" if engine is not None else "degraded",
        model_loaded=engine is not None and engine.model is not None,
        database_ok=db_ok,
        groq_configured=groq_configured(),
    )


@app.get("/api/v1/model/info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        name="icdas_mobilenet_cbam",
        num_classes=settings.num_classes,
        icdas_mode=settings.icdas_mode,
        image_size=settings.image_size,
        ordinal_regression=settings.ordinal_regression,
    )


def _run_prediction(
    content: bytes,
    filename: str,
    include_explainability: bool,
    user_id: str,
) -> PredictionResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_mb}MB limit",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    suffix = Path(filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG, PNG, BMP, or WEBP image.",
        )

    logger.info("Prediction request received (%s bytes, user=%s)", len(content), user_id)

    try:
        original, processed = engine.preprocess_upload(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {exc}") from exc

    try:
        result = engine.predict(processed)
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=503, detail="Model unavailable or prediction failed.") from exc

    try:
        action = get_clinical_action(result["icdas_grade"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    report_data = None
    try:
        report_data = generate_report(
            icdas_grade=result["icdas_grade"],
            confidence=result["confidence"],
            finding=action["finding"],
            urgency=action["urgency"],
        )
    except Exception:
        logger.exception("Groq failure during prediction")

    heatmap_b64 = overlay_b64 = contour_b64 = None
    heatmap_path = None
    if include_explainability:
        try:
            explain = engine.explain(processed, original, result["icdas_grade"])
            heatmap_b64 = explain["heatmap"]
            overlay_b64 = explain["overlay"]
            contour_b64 = explain["contour"]
            heatmap_path = _save_bytes(base64.b64decode(heatmap_b64), ".png")
        except Exception:
            logger.exception("Grad-CAM generation failed")

    image_path = _save_bytes(content, suffix or ".jpg")
    record_id = None
    db: Session | None = None
    try:
        db = SessionLocal()
        record = PredictionRecord(
            user_id=user_id or "anonymous",
            image_path=image_path,
            heatmap_path=heatmap_path,
            icdas_grade=result["icdas_grade"],
            confidence=result["confidence"],
            urgency=action["urgency"],
            finding=(report_data or {}).get("finding", action["finding"]),
            recommendation=(report_data or {}).get("recommendation", action["recommendation"]),
            report=(report_data or {}).get("report"),
            probabilities_json=json.dumps(result["probabilities"]),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        record_id = record.id
    except Exception:
        logger.exception("Database failure while storing prediction")
        if db is not None:
            db.rollback()
    finally:
        if db is not None:
            db.close()

    logger.info(
        "Prediction complete: grade=%s confidence=%.2f id=%s",
        result["icdas_grade"],
        result["confidence"],
        record_id,
    )

    return PredictionResponse(
        id=record_id,
        icdas_grade=result["icdas_grade"],
        confidence=result["confidence"],
        label=action["label"],
        action=action["action"],
        description=action["description"],
        finding=(report_data or {}).get("finding", action["finding"]),
        recommendation=(report_data or {}).get("recommendation", action["recommendation"]),
        urgency=(report_data or {}).get("urgency", action["urgency"]),
        probabilities=result["probabilities"],
        low_confidence=result["low_confidence"],
        low_confidence_message=result["low_confidence_message"],
        report=(report_data or {}).get("report"),
        heatmap_base64=heatmap_b64,
        overlay_base64=overlay_b64,
        contour_base64=contour_b64,
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    include_explainability: bool = True,
    x_user_id: str | None = Header(default="anonymous"),
):
    content = await file.read()
    return _run_prediction(
        content,
        file.filename or "upload.jpg",
        include_explainability,
        x_user_id or "anonymous",
    )


@app.post("/api/v1/report", response_model=ReportResponse)
async def create_report(data: ReportRequest):
    try:
        ai_result = generate_report(
            icdas_grade=data.icdas_grade,
            confidence=data.confidence,
            finding=data.finding,
            urgency=data.urgency,
            model_name=data.model_name or "MobileNetV3-Small + CBAM + ordinal regression",
        )
        return ReportResponse(
            icdas_grade=data.icdas_grade,
            confidence=data.confidence,
            finding=ai_result["finding"],
            recommendation=ai_result["recommendation"],
            urgency=ai_result["urgency"],
            report=ai_result["report"],
        )
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail="Report generation failed.") from exc


@app.get("/api/v1/history", response_model=list[HistoryItem])
async def list_history(limit: int = Query(50, ge=1, le=200)):
    db = SessionLocal()
    try:
        rows = (
            db.query(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
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
                has_image=bool(row.image_path),
                has_heatmap=bool(row.heatmap_path),
            )
            for row in rows
        ]
    except Exception as exc:
        logger.exception("Database failure while listing history")
        raise HTTPException(status_code=503, detail="Could not load prediction history.") from exc
    finally:
        db.close()


@app.get("/api/v1/history/{prediction_id}", response_model=HistoryDetail)
async def get_history_item(prediction_id: int):
    db = SessionLocal()
    try:
        row = db.get(PredictionRecord, prediction_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        probs = json.loads(row.probabilities_json) if row.probabilities_json else None
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
            has_image=bool(row.image_path),
            has_heatmap=bool(row.heatmap_path),
            image_base64=_encode_file(row.image_path),
            heatmap_base64=_encode_file(row.heatmap_path),
        )
    finally:
        db.close()


@app.get("/api/v1/stats", response_model=StatsResponse)
async def stats():
    db = SessionLocal()
    try:
        rows = db.query(PredictionRecord).all()
        if not rows:
            return StatsResponse(
                total_analyses=0,
                average_confidence=0.0,
                most_common_grade=None,
                high_severity_cases=0,
                grade_distribution={str(i): 0 for i in range(settings.num_classes)},
                confidence_buckets={"0-50": 0, "50-70": 0, "70-85": 0, "85-100": 0},
            )
        grades = [row.icdas_grade for row in rows]
        confs = [row.confidence for row in rows]
        dist = Counter(grades)
        buckets = {"0-50": 0, "50-70": 0, "70-85": 0, "85-100": 0}
        for conf in confs:
            if conf < 50:
                buckets["0-50"] += 1
            elif conf < 70:
                buckets["50-70"] += 1
            elif conf < 85:
                buckets["70-85"] += 1
            else:
                buckets["85-100"] += 1
        return StatsResponse(
            total_analyses=len(rows),
            average_confidence=round(sum(confs) / len(confs), 2),
            most_common_grade=max(dist, key=dist.get),
            high_severity_cases=sum(1 for g in grades if g >= 3),
            grade_distribution={str(i): int(dist.get(i, 0)) for i in range(settings.num_classes)},
            confidence_buckets=buckets,
        )
    except Exception as exc:
        logger.exception("Database failure while computing stats")
        raise HTTPException(status_code=503, detail="Could not load analytics.") from exc
    finally:
        db.close()
