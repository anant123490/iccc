"""Application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------
    # ML model configuration
    # ---------------------------------------------------------

    model_path: str = "../models/best.keras"
    deploy_model_path: str = "../models/deploy.keras"

    num_classes: int = 7
    image_size: int = 224

    # ICDAS classification mode
    icdas_mode: str = "0-6"

    # ---------------------------------------------------------
    # API configuration
    # ---------------------------------------------------------

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    max_upload_mb: int = 10

    # ---------------------------------------------------------
    # Groq configuration
    # ---------------------------------------------------------

    groq_api_key: str = Field(
        ...,
        validation_alias="GROQ_API_KEY",
    )

    # ---------------------------------------------------------
    # Environment configuration
    # ---------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ICDAS_",
        extra="ignore",
    )


# ---------------------------------------------------------
# Cached settings
# ---------------------------------------------------------

@lru_cache
def get_settings() -> Settings:
    return Settings()