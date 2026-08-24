"""
Application configuration for ICDAS 0-4.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = (
    Path(__file__).resolve().parents[1]
)

PROJECT_ROOT = (
    BACKEND_DIR.parent
)


# ============================================================
# SETTINGS
# ============================================================

class Settings(BaseSettings):

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model_path: str = str(
        PROJECT_ROOT
        / "models"
        / "deploy.keras"
    )

    deploy_model_path: str = str(
        PROJECT_ROOT
        / "models"
        / "deploy.keras"
    )

    # --------------------------------------------------------
    # ICDAS
    # --------------------------------------------------------

    num_classes: int = 5

    image_size: int = 224

    icdas_mode: str = "0-4"

    ordinal_regression: bool = False

    confidence_threshold: float = 0.55

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    use_roi_detection: bool = False

    use_clahe: bool = False

    use_specular_reduction: bool = False

    color_normalize: bool = False

    # --------------------------------------------------------
    # CORS
    # --------------------------------------------------------

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    max_upload_mb: int = 10

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    groq_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY",
    )

    groq_model: str = Field(
        default="openai/gpt-oss-20b",
        validation_alias="GROQ_MODEL",
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    database_url: str = Field(
        default=(
            f"sqlite:///"
            f"{(BACKEND_DIR / 'icdas_predictions.db').as_posix()}"
        ),
        validation_alias="DATABASE_URL",
    )

    jwt_secret: str = Field(
        default="",
        validation_alias="JWT_SECRET",
    )

    # --------------------------------------------------------
    # PYDANTIC
    # --------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=(
            str(BACKEND_DIR / ".env"),
            str(PROJECT_ROOT / ".env"),
        ),
        env_prefix="ICDAS_",
        populate_by_name=True,
        extra="ignore",
    )


# ============================================================
# SETTINGS FACTORY
# ============================================================

@lru_cache
def get_settings() -> Settings:

    return Settings()


# Global instance
settings = get_settings()


__all__ = [
    "Settings",
    "settings",
    "get_settings",
]