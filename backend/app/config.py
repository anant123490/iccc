"""
Application configuration for ICDAS 0–4 Dental Caries Detection Backend.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


# ============================================================
# SETTINGS
# ============================================================

class Settings(BaseSettings):
    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    # Always use the deployment model for inference.
    model_path: str = str(PROJECT_ROOT / "models" / "deploy.keras")
    deploy_model_path: str = str(PROJECT_ROOT / "models" / "deploy.keras")

    # --------------------------------------------------------
    # ICDAS CONFIGURATION
    # --------------------------------------------------------

    num_classes: int = 5
    image_size: int = 224
    icdas_mode: str = "0-4"

    # Your trained model is a 5-class SOFTMAX classifier.
    ordinal_regression: bool = False

    confidence_threshold: float = 0.50

    # --------------------------------------------------------
    # CORS CONFIGURATION
    # --------------------------------------------------------

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    max_upload_mb: int = 10

    # --------------------------------------------------------
    # GROQ CONFIGURATION
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
    # DATABASE CONFIGURATION
    # --------------------------------------------------------

    database_url: str = Field(
        default=f"sqlite:///{(BACKEND_DIR / 'icdas_predictions.db').as_posix()}",
        validation_alias="DATABASE_URL",
    )

    jwt_secret: str = Field(
        default="",
        validation_alias="JWT_SECRET",
    )

    # --------------------------------------------------------
    # PYDANTIC SETTINGS
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
# GLOBAL SETTINGS INSTANCE
# ============================================================

@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


# Global settings object imported across the backend.
settings = get_settings()

__all__ = ["Settings", "get_settings", "settings"]