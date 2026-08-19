"""Pydantic request/response schemas."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

AI_RESULT_NOTE = "AI result is not a final diagnosis."
DISCLAIMER = (
    "This tool is an AI decision-support / research prototype and is not "
    "a substitute for professional dental diagnosis."
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_ok: bool = True
    groq_configured: bool = False
    disclaimer: str = DISCLAIMER


class ModelInfoResponse(BaseModel):
    name: str
    num_classes: int
    icdas_mode: str
    image_size: int
    architecture: str = "MobileNetV3Small + CBAM + ordinal regression"
    ordinal_regression: bool = True


class PredictionResponse(BaseModel):
    id: Optional[int] = None
    icdas_grade: int = Field(..., ge=0, le=4)
    confidence: float
    label: str
    action: str
    description: str
    finding: str
    recommendation: str
    urgency: str
    probabilities: Dict[str, float]
    low_confidence: bool = False
    low_confidence_message: Optional[str] = None
    report: Optional[str] = None
    ai_result_note: str = AI_RESULT_NOTE
    heatmap_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    contour_base64: Optional[str] = None
    disclaimer: str = DISCLAIMER


class ReportRequest(BaseModel):
    icdas_grade: int = Field(..., ge=0, le=4)
    confidence: float = Field(..., ge=0, le=100)
    finding: Optional[str] = None
    urgency: Optional[str] = None
    model_name: Optional[str] = "MobileNetV3-Small + CBAM + ordinal regression"


class ReportResponse(BaseModel):
    icdas_grade: int = Field(..., ge=0, le=4)
    confidence: float
    finding: str
    recommendation: str
    urgency: str
    report: str


class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    icdas_grade: int
    confidence: float
    urgency: str
    finding: Optional[str] = None
    has_image: bool = False
    has_heatmap: bool = False


class HistoryDetail(HistoryItem):
    recommendation: Optional[str] = None
    report: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    image_base64: Optional[str] = None
    heatmap_base64: Optional[str] = None


class StatsResponse(BaseModel):
    total_analyses: int
    average_confidence: float
    most_common_grade: Optional[int] = None
    high_severity_cases: int
    grade_distribution: Dict[str, int]
    confidence_buckets: Dict[str, int]
