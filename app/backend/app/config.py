"""
Application configuration for ICDAS 0-4.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
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


def _resolve_project_root() -> Path:
    """Repo root locally (`…/app/backend`); `/app` in Docker."""
    docker_root = BACKEND_DIR.parent
    if (docker_root / "ml").is_dir() and (docker_root / "models").is_dir():
        return docker_root
    repo_root = BACKEND_DIR.parent.parent
    if (repo_root / "ml").is_dir():
        return repo_root
    return docker_root


PROJECT_ROOT = _resolve_project_root()
DEFAULT_SQLITE_NAME = "icdas_predictions.db"


def resolve_database_url(url: str, base_dir: Path | None = None) -> str:
    """Make relative SQLite paths independent of process CWD.

    Relative sqlite:///./file.db URLs resolve against the project root.
    Absolute SQLite paths and non-SQLite URLs are left unchanged.
    """
    from sqlalchemy.engine.url import make_url

    root = Path(base_dir or PROJECT_ROOT).resolve()
    raw = (url or "").strip()
    if not raw:
        return f"sqlite:///{(root / DEFAULT_SQLITE_NAME).as_posix()}"
    parsed = make_url(raw)
    dialect = parsed.drivername.split("+", 1)[0].lower()
    if dialect != "sqlite":
        return raw
    database = parsed.database
    if not database or database == ":memory:":
        return raw
    db_path = Path(database)
    if not db_path.is_absolute():
        db_path = (root / db_path).resolve()
    else:
        db_path = db_path.resolve()
    return str(parsed.set(database=db_path.as_posix()))


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
        / "icdas"
        / "current"
        / "deploy.keras"
    )

    deploy_model_path: str = str(
        PROJECT_ROOT
        / "models"
        / "icdas"
        / "current"
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
        "http://localhost:8502",
        "http://127.0.0.1:8502",
        "http://localhost:8503",
        "http://127.0.0.1:8503",
    ]

    admin_password: str = Field(
        default="changeme",
        validation_alias="ICDAS_ADMIN_PASSWORD",
    )

    allow_icdas_train: bool = Field(
        default=False,
        validation_alias="ICDAS_ALLOW_TRAIN",
    )

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
            f"{(PROJECT_ROOT / DEFAULT_SQLITE_NAME).as_posix()}"
        ),
        validation_alias="DATABASE_URL",
    )

    jwt_secret: str = Field(
        default="",
        validation_alias="JWT_SECRET",
    )

    @field_validator("database_url", mode="after")
    @classmethod
    def _resolve_sqlite_database_url(cls, value: str) -> str:
        return resolve_database_url(value, PROJECT_ROOT)

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
    "PROJECT_ROOT",
    "BACKEND_DIR",
    "resolve_database_url",
]