"""Tests for safe per-photo deactivation feature in Admin training dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.main import app
from app.portal_auth import issue_admin_token
from app.portal_db import (
    DatasetVersion,
    Patient,
    TrainingImage,
    TrainingLabel,
    Visit,
)
from app.training_workflow import (
    STATUS_UNIQUE,
    _eligible_label_query,
    build_dataset_version,
    classify_duplicate,
    deactivate_training_image,
    dataset_inventory,
    list_training_images,
    training_image_detail,
)


@pytest.fixture
def db_session(tmp_path: Path):
    db_file = tmp_path / "test_deactivation.db"
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def _create_dummy_jpeg(path: Path, val: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((20, 20, 3), (val * 23) % 256, dtype=np.uint8)
    img[0, 0, 0] = (val * 17) % 256
    img[1, 1, 1] = (val * 31) % 256
    img[2, 2, 2] = (val * 47) % 256
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        path.write_bytes(buf.tobytes())


def _create_sample_training_photo(
    db, filename: str, content_hash: str, crops_with_grades: list[int | None], tmp_path: Path | None = None
) -> TrainingImage:
    if tmp_path is not None:
        img_path = tmp_path / filename
        _create_dummy_jpeg(img_path, val=hash(filename) % 200)
        img_path_str = str(img_path)
    else:
        img_path_str = f"/fake/path/{filename}"

    img = TrainingImage(
        filename=filename,
        path=img_path_str,
        overlay_path=img_path_str,
        duplicate_status=STATUS_UNIQUE,
        boxes_verified=True,
        exclude_from_dataset=False,
        is_active=True,
        content_hash=content_hash,
    )
    db.add(img)
    db.flush()

    for idx, g in enumerate(crops_with_grades):
        if tmp_path is not None:
            cpath = tmp_path / f"crop_{img.id}_{idx}.jpg"
            _create_dummy_jpeg(cpath, val=img.id * 100 + idx + 1)
            cpath_str = str(cpath)
        else:
            cpath_str = f"/fake/crop_{img.id}_{idx}.jpg"

        lab = TrainingLabel(
            image_id=img.id,
            crop_path=cpath_str,
            grade=g,
            box_verified=True,
            active=True,
            skipped=False,
            crop_duplicate_status=STATUS_UNIQUE,
            crop_hash=f"hash_{img.id}_{idx}",
        )
        db.add(lab)
    db.commit()
    db.refresh(img)
    return img


def test_active_photo_eligibility(db_session, tmp_path):
    img = _create_sample_training_photo(db_session, "active1.jpg", "hash_act_1", [0, 1, 2], tmp_path)
    eligible = list(_eligible_label_query(db_session).all())
    eligible_img_ids = {lab.image_id for lab in eligible}

    assert img.id in eligible_img_ids
    assert len(eligible) == 3


def test_deactivated_photo_and_crops_ineligible(db_session, tmp_path):
    img = _create_sample_training_photo(db_session, "deact1.jpg", "hash_deact_1", [0, 1, 2], tmp_path)

    # Deactivate the photo
    res = deactivate_training_image(db_session, img.id)
    assert res["is_active"] is False
    assert res["deactivated"] is True

    # Check eligibility
    eligible = list(_eligible_label_query(db_session).all())
    eligible_img_ids = {lab.image_id for lab in eligible}
    assert img.id not in eligible_img_ids

    # Check image detail & list
    detail = training_image_detail(db_session, img.id)
    assert detail["is_active"] is False

    listing = list_training_images(db_session)
    item = next(i for i in listing["images"] if i["training_image_id"] == img.id)
    assert item["is_active"] is False


def test_deactivated_photo_labels_excluded_from_new_build(db_session, monkeypatch, tmp_path: Path):
    # Patch directory paths for build
    import app.training_workflow as tw

    vdir = tmp_path / "versions"
    vdir.mkdir()
    monkeypatch.setattr(tw, "TRAIN_VERSIONS", vdir)

    # Create active photos with enough crops for minimum build
    img1 = _create_sample_training_photo(db_session, "p1.jpg", "h1", [0, 1, 2, 3, 4], tmp_path)
    img2 = _create_sample_training_photo(db_session, "p2.jpg", "h2", [0, 1, 2, 3, 4], tmp_path)

    # Deactivate img2
    deactivate_training_image(db_session, img2.id)

    # Build dataset
    out = build_dataset_version(db_session)
    assert out["name"] == "v1"

    man_file = vdir / "v1" / "manifest.csv"
    assert man_file.exists()

    with man_file.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    included_img2 = [
        row for row in rows
        if row.get("source_image_id") == str(img2.id) and row.get("status") == "INCLUDED"
    ]
    assert len(included_img2) == 0


def test_existing_dataset_versions_remain_untouched(db_session, tmp_path):
    ver = DatasetVersion(
        name="v1",
        path="/fake/versions/v1",
        status="COMPLETED",
        n_train=10,
        n_valid=2,
        n_test=2,
        version_number=1,
    )
    db_session.add(ver)
    db_session.commit()

    img = _create_sample_training_photo(db_session, "p_version.jpg", "h_ver", [0], tmp_path)
    deactivate_training_image(db_session, img.id)

    db_ver = db_session.query(DatasetVersion).filter(DatasetVersion.id == ver.id).first()
    assert db_ver is not None
    assert db_ver.name == "v1"
    assert db_ver.status == "COMPLETED"


def test_deactivation_is_idempotent(db_session, tmp_path):
    img = _create_sample_training_photo(db_session, "idem.jpg", "h_idem", [0, 1], tmp_path)

    res1 = deactivate_training_image(db_session, img.id)
    assert res1["is_active"] is False

    # Second call must be safe and return success
    res2 = deactivate_training_image(db_session, img.id)
    assert res2["is_active"] is False


def test_patient_records_unaffected(db_session, tmp_path):
    patient = Patient(public_id="P100", name="John Doe")
    db_session.add(patient)
    db_session.flush()

    visit = Visit(patient_id=patient.id, visit_date="2026-09-01")
    db_session.add(visit)
    db_session.commit()

    img = _create_sample_training_photo(db_session, "p_train.jpg", "h_p_train", [0], tmp_path)
    deactivate_training_image(db_session, img.id)

    p_check = db_session.query(Patient).filter(Patient.id == patient.id).first()
    v_check = db_session.query(Visit).filter(Visit.id == visit.id).first()
    assert p_check is not None
    assert p_check.name == "John Doe"
    assert v_check is not None


def test_duplicate_detection_excludes_deactivated_photos(db_session, tmp_path):
    img1 = _create_sample_training_photo(db_session, "canon.jpg", "dup_hash_1", [0], tmp_path)

    # Deactivate canon image
    deactivate_training_image(db_session, img1.id)

    # Classify a new photo with identical content hash
    status, canon_id = classify_duplicate(db_session, "dup_hash_1", phash=None)
    # Since canon is deactivated, new photo is treated as UNIQUE
    assert status == STATUS_UNIQUE
    assert canon_id is None


def test_dataset_readiness(db_session, tmp_path):
    img1 = _create_sample_training_photo(db_session, "r1.jpg", "hr1", [0, 1, 2, 3, 4], tmp_path)
    inv1 = dataset_inventory(db_session)
    assert inv1["crops"]["eligible"] == 5

    deactivate_training_image(db_session, img1.id)

    inv2 = dataset_inventory(db_session)
    assert inv2["crops"]["eligible"] == 0


@pytest.mark.asyncio
async def test_deactivate_api_endpoint(db_session, tmp_path):
    token = issue_admin_token()
    transport = ASGITransport(app=app)

    img = _create_sample_training_photo(db_session, "api_photo.jpg", "h_api", [0], tmp_path)
    img_id = img.id

    from app.database import SessionLocal, get_db

    def _override_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            res = await ac.post(
                f"/api/v1/admin/training/photos/{img_id}/deactivate",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert res.status_code == 200
            body = res.json()
            assert body["training_image_id"] == img_id
            assert body["is_active"] is False

            db_check = db_session.query(TrainingImage).filter(TrainingImage.id == img_id).first()
            assert db_check.is_active is False
    finally:
        app.dependency_overrides.clear()
