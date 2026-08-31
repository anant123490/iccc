"""Tooth-crop exact-duplicate protection (no YOLO/ICDAS training)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.crop_identity import (
    CONFLICT_MESSAGE,
    CROP_CONFLICT,
    CROP_DUP_EXACT,
    CROP_UNIQUE,
    apply_crop_identity,
    crop_content_sha256,
)
from app.portal_db import DatasetVersion, TrainingJob, TrainingLabel
from app.training_workflow import (
    INCLUDED,
    build_dataset_version,
    dataset_inventory,
    save_training_boxes,
    validate_built_dataset,
    validate_dataset_for_train,
)

from test_dataset_hardening import (
    _add_crop,
    _add_image,
    _patch_wf,
    _session,
)


def test_same_crop_bytes_second_is_duplicate(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    a = _add_crop(db, dirs, img, grade=1, idx=0, pixel_seed=7)
    b = _add_crop(db, dirs, img, grade=1, idx=1, pixel_seed=7)
    db.refresh(a)
    db.refresh(b)
    assert a.crop_hash == b.crop_hash
    assert a.crop_duplicate_status == CROP_UNIQUE
    assert b.crop_duplicate_status == CROP_DUP_EXACT
    assert b.duplicate_of_label_id == a.id
    inv = dataset_inventory(db)
    assert inv["crops"]["eligible"] == 1
    assert inv["crops"]["exact_duplicates"] == 1


def test_same_crop_in_two_photos_one_eligible(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img_a = _add_image(db, dirs, name="a.jpg")
    img_b = _add_image(db, dirs, name="b.jpg")
    a = _add_crop(db, dirs, img_a, grade=2, idx=0, pixel_seed=9)
    b = _add_crop(db, dirs, img_b, grade=2, idx=0, pixel_seed=9)
    db.refresh(a)
    db.refresh(b)
    assert a.crop_hash == b.crop_hash
    assert {a.crop_duplicate_status, b.crop_duplicate_status} == {CROP_UNIQUE, CROP_DUP_EXACT}
    inv = dataset_inventory(db)
    assert inv["crops"]["eligible"] == 1
    assert inv["crops"]["labeled"] == 2


def test_duplicate_same_icdas_keeps_one_sample(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    _add_crop(db, dirs, img, grade=0, idx=0, pixel_seed=3)
    _add_crop(db, dirs, img, grade=0, idx=1, pixel_seed=3)
    inv = dataset_inventory(db)
    assert inv["crops"]["eligible"] == 1
    assert inv["class_counts"]["0"] == 1


def test_duplicate_conflicting_icdas_excluded(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    a = _add_crop(db, dirs, img, grade=1, idx=0, pixel_seed=4)
    b = _add_crop(db, dirs, img, grade=3, idx=1, pixel_seed=4)
    db.refresh(a)
    db.refresh(b)
    assert a.crop_duplicate_status == CROP_CONFLICT
    assert b.crop_duplicate_status == CROP_CONFLICT
    assert a.grade == 1 and b.grade == 3
    inv = dataset_inventory(db)
    assert inv["crops"]["eligible"] == 0
    assert inv["crops"]["conflicts"] == 2
    assert CONFLICT_MESSAGE in (inv.get("crop_conflict_message") or "")


def test_different_crop_images_both_eligible(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    a = _add_crop(db, dirs, img, grade=0, idx=0)
    b = _add_crop(db, dirs, img, grade=1, idx=1)
    db.refresh(a)
    db.refresh(b)
    assert a.crop_hash != b.crop_hash
    assert a.crop_duplicate_status == CROP_UNIQUE
    assert b.crop_duplicate_status == CROP_UNIQUE
    inv = dataset_inventory(db)
    assert inv["crops"]["eligible"] == 2


def test_inactive_regenerated_crop_not_extra_sample(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs, verified=False)
    old = _add_crop(db, dirs, img, grade=1, idx=0, verified=False, pixel_seed=5)
    save_training_boxes(
        db,
        img.id,
        [{"label_id": old.id, "x1": 12, "y1": 12, "x2": 70, "y2": 72}],
        [],
    )
    db.refresh(old)
    assert old.active is True
    ghost = _add_crop(db, dirs, img, grade=1, idx=1, verified=False, active=False, pixel_seed=5)
    db.refresh(ghost)
    assert ghost.active is False
    apply_crop_identity(db, old)
    apply_crop_identity(db, ghost)
    db.commit()
    actives = [l for l in db.query(TrainingLabel).all() if l.active is not False]
    hashes = [l.crop_hash for l in actives if l.crop_hash]
    assert len(hashes) == len(set(hashes)) or all(
        l.crop_duplicate_status != CROP_UNIQUE or l.active is not False for l in db.query(TrainingLabel)
    )
    unique_active = [
        l
        for l in db.query(TrainingLabel).all()
        if l.active is not False and (l.crop_duplicate_status or CROP_UNIQUE) == CROP_UNIQUE
    ]
    assert len(unique_active) == 1


def test_build_dataset_has_unique_crop_hashes(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    for i, g in enumerate((0, 1, 2, 3, 4)):
        _add_crop(db, dirs, img, grade=g, idx=i)
    img2 = _add_image(db, dirs, name="dupphoto.jpg")
    _add_crop(db, dirs, img2, grade=0, idx=0, pixel_seed=img.id * 100 + 0)
    jobs_before = db.query(TrainingJob).count()
    out = build_dataset_version(db)
    assert db.query(TrainingJob).count() == jobs_before
    man = Path(out["path"]) / "manifest.csv"
    included = []
    with man.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == INCLUDED:
                included.append(row["crop_hash"])
    assert included
    assert len(included) == len(set(included))
    check = validate_built_dataset(Path(out["path"]))
    assert not any("Duplicate crop hash" in x for x in check["issues"])


def test_build_does_not_overwrite_existing_version(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs, name="a.jpg")
    for g in range(5):
        _add_crop(db, dirs, img, grade=g, idx=g)
    v1 = build_dataset_version(db)
    v1_path = Path(v1["path"])
    marker = v1_path / "keep_me.txt"
    marker.write_text("v1-intact", encoding="utf-8")
    _add_crop(db, dirs, img, grade=0, idx=20)
    v2 = build_dataset_version(db)
    assert v1["name"] == "v1" and v2["name"] == "v2"
    assert marker.read_text(encoding="utf-8") == "v1-intact"
    assert db.query(DatasetVersion).count() == 2


def test_identity_refresh_does_not_change_grades(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    labs = [_add_crop(db, dirs, img, grade=g, idx=g) for g in range(5)]
    before = [(lab.id, lab.grade) for lab in labs]
    dataset_inventory(db)
    after = [
        (row.id, row.grade)
        for row in db.query(TrainingLabel).order_by(TrainingLabel.id.asc()).all()
        if row.id in {lab.id for lab in labs}
    ]
    assert before == after


def test_train_blocked_on_unresolved_crop_conflict(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    for i, g in enumerate((0, 1, 2, 3, 4)):
        _add_crop(db, dirs, img, grade=g, idx=i)
    build_dataset_version(db)
    img2 = _add_image(db, dirs, name="c.jpg")
    _add_crop(db, dirs, img2, grade=1, idx=0, pixel_seed=999)
    _add_crop(db, dirs, img2, grade=3, idx=1, pixel_seed=999)
    with pytest.raises(ValueError, match="conflicting ICDAS labels"):
        validate_dataset_for_train(db, None)


def test_normalized_hash_matches_decoded_pixels():
    rgb = __import__("numpy").full((16, 16, 3), 90, dtype="uint8")
    rgb[2, 2] = (1, 2, 3)
    h1 = crop_content_sha256(rgb)
    h2 = crop_content_sha256(rgb.copy())
    assert h1 == h2
    assert len(h1) == 64
