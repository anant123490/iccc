"""Application configuration."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    model_path: str = "../models/best.keras"
    deploy_model_path: str = "../models/deploy.keras"
    num_classes: int = 7
    image_size: int = 224
    icdas_mode: str = "0-6"  # or "0-4"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    max_upload_mb: int = 10

    class Config:
        env_file = ".env"
        env_prefix = "ICDAS_"


@lru_cache
def get_settings() -> Settings:
    return Settings()
