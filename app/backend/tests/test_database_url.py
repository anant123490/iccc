"""SQLite DATABASE_URL must not depend on process CWD."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine.url import make_url

from app.config import PROJECT_ROOT, get_settings, resolve_database_url


def test_relative_sqlite_url_is_project_root_not_cwd(tmp_path, monkeypatch):
    expected = (PROJECT_ROOT / "icdas_predictions.db").resolve()
    monkeypatch.chdir(tmp_path)
    from_tmp = resolve_database_url("sqlite:///./icdas_predictions.db", PROJECT_ROOT)
    monkeypatch.chdir(PROJECT_ROOT / "app" / "backend")
    from_backend = resolve_database_url("sqlite:///./icdas_predictions.db", PROJECT_ROOT)
    assert from_tmp == from_backend
    assert Path(make_url(from_tmp).database).resolve() == expected


def test_absolute_sqlite_and_postgres_urls_unchanged(tmp_path):
    abs_db = (tmp_path / "other.db").resolve()
    url = f"sqlite:///{abs_db.as_posix()}"
    resolved = resolve_database_url(url, PROJECT_ROOT)
    assert Path(make_url(resolved).database).resolve() == abs_db
    pg = "postgresql+psycopg2://user:pass@localhost:5432/icdas"
    assert resolve_database_url(pg, PROJECT_ROOT) == pg
    assert resolve_database_url("sqlite:///:memory:", PROJECT_ROOT) == "sqlite:///:memory:"


def test_settings_resolve_same_sqlite_from_backend_and_root_cwd(monkeypatch):
    expected = (PROJECT_ROOT / "icdas_predictions.db").resolve()
    monkeypatch.chdir(PROJECT_ROOT)
    get_settings.cache_clear()
    a = get_settings().database_url
    monkeypatch.chdir(PROJECT_ROOT / "app" / "backend")
    get_settings.cache_clear()
    b = get_settings().database_url
    assert a == b
    assert Path(make_url(a).database).resolve() == expected
    get_settings.cache_clear()
