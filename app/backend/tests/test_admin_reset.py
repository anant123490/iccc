"""Admin training reset: confirmation, scoped delete, no model/weight deletion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.db_models import PredictionRecord
from app.main import app
from app.portal_db import (
    DatasetVersion,
    ModelVersion,
    Patient,
    TrainingImage,
    TrainingJob,
    TrainingLabel,
)
from app.training_reset import (
    RESET_CONFIRMATION_TEXT,
    _empty_allowed_dir,
    execute_training_reset,
    reset_plan,
    validate_reset_request,
)
from app.training_workflow import dataset_inventory, labeling_queue

STREAMLIT_ROOT = Path(__file__).resolve().parents[2] / "frontend" / "streamlit"
sys.path.insert(0, str(STREAMLIT_ROOT))
from shared.admin_reset import RESET_CONFIRMATION_TEXT as UI_PHRASE  # noqa: E402
from shared.admin_workflow import merge_pending_nav  # noqa: E402
from shared.upload_state import clear_admin_training_ui_state  # noqa: E402


PHRASE = RESET_CONFIRMATION_TEXT


def _patch_training_dirs(monkeypatch, tmp_path: Path):
    import app.training_reset as tr

    root = tmp_path / "training"
    uploads = root / "uploads"
    detected = root / "detected"
    labeled = root / "labeled"
    versions = root / "versions"
    for d in (uploads, detected, labeled, versions):
        d.mkdir(parents=True)
    monkeypatch.setattr(tr, "TRAINING_ROOT", root)
    monkeypatch.setattr(tr, "TRAIN_UPLOADS", uploads)
    monkeypatch.setattr(tr, "TRAIN_DETECTED", detected)
    monkeypatch.setattr(tr, "TRAIN_LABELED", labeled)
    monkeypatch.setattr(tr, "TRAIN_VERSIONS", versions)
    return {
        "root": root,
        "uploads": uploads,
        "detected": detected,
        "labeled": labeled,
        "versions": versions,
    }


def _session(tmp_path: Path):
    import app.db_models  # noqa: F401
    import app.portal_db  # noqa: F401

    engine = create_engine(f"sqlite:///{tmp_path / 'reset.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed_training(db, dirs: dict, *, with_job: bool = True) -> None:
    (dirs["uploads"] / "photo.jpg").write_bytes(b"fake-jpg")
    crop = dirs["detected"] / "1" / "tooth_001.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"crop")
    (dirs["labeled"] / "0").mkdir(parents=True)
    (dirs["labeled"] / "0" / "a.jpg").write_bytes(b"lab")
    man = dirs["versions"] / "v1"
    man.mkdir()
    (man / "manifest.csv").write_text("split,path\n", encoding="utf-8")
    img = TrainingImage(
        filename="photo.jpg",
        path=str(dirs["uploads"] / "photo.jpg"),
        overlay_path=str(dirs["detected"] / "1" / "overlay.jpg"),
        n_crops=1,
        content_hash="abc",
        duplicate_status="UNIQUE",
        boxes_verified=True,
    )
    db.add(img)
    db.flush()
    db.add(
        TrainingLabel(
            image_id=img.id,
            crop_path=str(crop),
            grade=1,
            box_verified=True,
            active=True,
            skipped=False,
            x2=8,
            y2=8,
        )
    )
    db.add(
        DatasetVersion(
            name="v1",
            path=str(man),
            status="READY",
            n_train=1,
            version_number=1,
        )
    )
    if with_job:
        db.add(TrainingJob(status="blocked", message="disabled", log_text="off"))
    db.commit()


def test_confirmation_phrase_matches_admin_ui():
    assert UI_PHRASE == RESET_CONFIRMATION_TEXT
    assert "permanently deletes" in PHRASE


def test_reset_confirmation_required():
    with pytest.raises(ValueError, match="required"):
        validate_reset_request("dataset", False, PHRASE)
    with pytest.raises(ValueError, match="Confirmation text"):
        validate_reset_request("dataset", True, "delete everything")
    with pytest.raises(ValueError, match="scope"):
        validate_reset_request("everything", True, PHRASE)
    assert validate_reset_request("dataset", True, PHRASE) == "dataset"


def test_reset_plan_does_not_include_models_or_source():
    plan = reset_plan("dataset")
    joined = " ".join(plan["will_not_delete"])
    assert "model_versions" in joined
    assert "training_jobs" not in plan["database_tables"]
    full = reset_plan("full")
    assert "training_jobs" in full["database_tables"]
    for item in plan["paths"]:
        assert "data" in item["path"].replace("\\", "/") or "training" in item["path"]


def test_refuses_path_outside_training_root(monkeypatch, tmp_path):
    _patch_training_dirs(monkeypatch, tmp_path)
    outside = tmp_path / "models" / "detection" / "best.pt"
    outside.parent.mkdir(parents=True)
    outside.write_text("weights", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing"):
        _empty_allowed_dir(outside)
    assert outside.read_text(encoding="utf-8") == "weights"


def test_dataset_reset_clears_files_labels_metrics(monkeypatch, tmp_path):
    dirs = _patch_training_dirs(monkeypatch, tmp_path)
    db = _session(tmp_path)
    db.add(Patient(public_id="PKEEP", name="Keep Me"))
    db.add(PredictionRecord(icdas_grade=2, confidence=0.9))
    db.add(ModelVersion(name="Tooth Detector V2", kind="detection", path="models/keep.pt"))
    weight = tmp_path / "models" / "detection" / "tooth_detector_v2" / "weights" / "best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"YOLO-WEIGHTS")
    source = Path(__file__).resolve()
    _seed_training(db, dirs)

    out = execute_training_reset(db, "dataset", True, PHRASE)

    assert list(dirs["uploads"].iterdir()) == []
    assert list(dirs["detected"].iterdir()) == []
    assert list(dirs["labeled"].iterdir()) == []
    assert list(dirs["versions"].iterdir()) == []
    assert db.query(TrainingImage).count() == 0
    assert db.query(TrainingLabel).count() == 0
    assert db.query(DatasetVersion).count() == 0
    assert db.query(TrainingJob).count() == 1
    assert db.query(Patient).count() == 1
    assert db.query(PredictionRecord).count() == 1
    assert db.query(ModelVersion).count() == 1
    assert weight.read_bytes() == b"YOLO-WEIGHTS"
    assert source.exists()
    assert out["uploads"] == 0
    assert out["duplicates"] == 0
    assert out["crops"] == 0
    assert out["labeled"] == 0
    assert out["unlabeled"] == 0
    assert out["dataset"] == "NOT READY"
    assert out["dataset_ready"] is False
    assert out["training"] == "DISABLED"
    assert out["launched"] is False
    inv = dataset_inventory(db)
    assert inv["dataset_ready"] is False
    assert inv["status"] == "NOT READY"


def test_full_reset_also_clears_jobs_not_weights(monkeypatch, tmp_path):
    dirs = _patch_training_dirs(monkeypatch, tmp_path)
    db = _session(tmp_path)
    _seed_training(db, dirs)
    weight = tmp_path / "models" / "icdas" / "v9" / "deploy.keras"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"KERAS")
    execute_training_reset(db, "full", True, PHRASE)
    assert db.query(TrainingJob).count() == 0
    assert weight.read_bytes() == b"KERAS"


def test_reset_is_idempotent(monkeypatch, tmp_path):
    dirs = _patch_training_dirs(monkeypatch, tmp_path)
    db = _session(tmp_path)
    _seed_training(db, dirs)
    execute_training_reset(db, "full", True, PHRASE)
    again = execute_training_reset(db, "full", True, PHRASE)
    assert again["uploads"] == 0
    assert again["ok"] is True


def test_upload_and_labeling_work_after_reset(monkeypatch, tmp_path):
    dirs = _patch_training_dirs(monkeypatch, tmp_path)
    db = _session(tmp_path)
    _seed_training(db, dirs)
    execute_training_reset(db, "dataset", True, PHRASE)
    q0 = labeling_queue(db)
    assert q0.get("current") is None

    dest = dirs["uploads"] / "new.jpg"
    dest.write_bytes(b"new")
    img = TrainingImage(
        filename="new.jpg",
        path=str(dest),
        n_crops=1,
        duplicate_status="UNIQUE",
        boxes_verified=True,
    )
    db.add(img)
    db.flush()
    crop = dirs["detected"] / "9" / "tooth_001.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"c")
    db.add(
        TrainingLabel(
            image_id=img.id,
            crop_path=str(crop),
            grade=None,
            box_verified=True,
            active=True,
            skipped=False,
            x2=4,
            y2=4,
        )
    )
    db.commit()
    q1 = labeling_queue(db)
    assert q1["current"]["filename"] == "new.jpg"
    assert q1["current"]["grade"] is None
    assert q1["next_enabled"] is False


def test_admin_navigation_and_uploader_state_after_reset():
    state = {
        "admin_page": "Dashboard",
        "train_up": {"count": 3},
        "last_upload_names": ["a.jpg"],
        "label_id": 99,
        "label_resume": True,
        "job": {"status": "blocked"},
        "training_uploader_nonce": 4,
    }
    clear_admin_training_ui_state(state)
    assert "train_up" not in state
    assert "label_id" not in state
    assert state["admin_page"] == "Dashboard"
    assert merge_pending_nav(state["admin_page"], None) == "Dashboard"
    assert state["training_uploader_nonce"] == 5


@pytest.mark.asyncio
async def test_http_reset_requires_auth_and_confirmation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.post(
            "/api/v1/admin/training/reset",
            json={"scope": "dataset", "confirm": True, "confirmation_text": PHRASE},
        )
        assert denied.status_code == 401
        login = await client.post("/api/v1/admin/login", json={"password": "changeme"})
        token = login.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        bad = await client.post(
            "/api/v1/admin/training/reset",
            json={"scope": "dataset", "confirm": False, "confirmation_text": PHRASE},
            headers=headers,
        )
        assert bad.status_code == 400
        assert "confirm" in (bad.json().get("detail") or "").lower()
        plan = await client.get("/api/v1/admin/training/reset/plan", headers=headers)
        assert plan.status_code == 200
        assert "training_images" in plan.json()["database_tables"]
