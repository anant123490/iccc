"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_path: str = str(PROJECT_ROOT / "models" / "best.keras")
    deploy_model_path: str = str(PROJECT_ROOT / "models" / "deploy.keras")

    num_classes: int = 5
    image_size: int = 224
    icdas_mode: str = "0-4"
    ordinal_regression: bool = True
    confidence_threshold: float = 0.55

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]
    max_upload_mb: int = 10

    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-20b", validation_alias="GROQ_MODEL")

    database_url: str = Field(
        default=f"sqlite:///{(BACKEND_DIR / 'icdas_predictions.db').as_posix()}",
        validation_alias="DATABASE_URL",
    )
    jwt_secret: str = Field(default="", validation_alias="JWT_SECRET")

    model_config = SettingsConfigDict(
        env_file=(
            str(BACKEND_DIR / ".env"),
            str(BACKEND_DIR.parent / ".env"),
        ),
        env_prefix="ICDAS_",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
