"""Pydantic request/response schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    disclaimer: str = (
        "This tool is for clinical decision support and is not "
        "a substitute for professional diagnosis."
    )


class ModelInfoResponse(BaseModel):
    name: str
    num_classes: int
    icdas_mode: str
    image_size: int
    architecture: str = "MobileNetV3Small + CBAM"
    ordinal_regression: bool = True


AI_RESULT_NOTE = "AI result is not final diagnosis."


class PredictionResponse(BaseModel):
    icdas_grade: int = Field(..., ge=0, le=6)
    confidence: float
    label: str
    action: str
    description: str
    finding: str
    recommendation: str
    urgency: str
    probabilities: List[float]
    ai_result_note: str = AI_RESULT_NOTE
    heatmap_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    contour_base64: Optional[str] = None
    disclaimer: str = (
        "This tool is for clinical decision support and is not "
        "a substitute for professional diagnosis."
    )


# ---------------------------------------------------------
# AI REPORT SCHEMAS
# ---------------------------------------------------------

class ReportRequest(BaseModel):
    icdas_grade: int = Field(..., ge=0, le=6)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ReportResponse(BaseModel):
    report: str