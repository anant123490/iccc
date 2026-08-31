"""Dataset build, crop, and ICDAS label hardening (no YOLO/ICDAS training)."""

from __future__ import annotations

from pathlib import Path


import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.portal_db import DatasetVersion, TrainingImage, TrainingJob, TrainingLabel
from app.portal_routes import LabelIn
from app.tooth_detector import box_iou_xyxy, validate_xyxy_box
from app.training_workflow import (
    STATUS_EXACT,
    STATUS_UNIQUE,
    build_dataset_version,
    dataset_inventory,
    labeling_queue,
    save_icdas_label,
    save_training_boxes,
    split_source_images,
    validate_dataset_for_train,
)
from app.crop_identity import apply_crop_identity


def _jpg_bytes(w=96, h=96, seed=0) -> bytes:
    rgb = np.full((h, w, 3), 70, dtype=np.uint8)
    block = max(1, min(w, h) // 8)
    for i in range(8):
        for j in range(8):
            on = ((seed >> ((i + 3 * j) % 16)) ^ i ^ j) & 1
            rgb[i * block : (i + 1) * block, j * block : (j + 1) * block] = (
                255 if on else 20,
                (seed * 13 + i * 17) % 256,
                (seed * 29 + j * 19) % 256,
            )
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _write_jpg(path: Path, w=96, h=96, seed=0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_jpg_bytes(w, h, seed))


def _patch_wf(monkeypatch, tmp_path: Path) -> dict:
    import app.training_workflow as tw

    root = tmp_path / "training"
    dirs = {
        "uploads": root / "uploads",
        "detected": root / "detected",
        "labeled": root / "labeled",
        "versions": root / "versions",
    }
    for d in dirs.values():
        d.mkdir(parents=True)
    monkeypatch.setattr(tw, "TRAIN_UPLOADS", dirs["uploads"])
    monkeypatch.setattr(tw, "TRAIN_DETECTED", dirs["detected"])
    monkeypatch.setattr(tw, "TRAIN_LABELED", dirs["labeled"])
    monkeypatch.setattr(tw, "TRAIN_VERSIONS", dirs["versions"])
    return dirs


def _session(tmp_path: Path):
    import app.db_models  # noqa: F401
    import app.portal_db  # noqa: F401

    engine = create_engine(f"sqlite:///{tmp_path / 'ds.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _add_image(db, dirs, *, unique=True, verified=True, name="p.jpg") -> TrainingImage:
    dest = dirs["uploads"] / name
    _write_jpg(dest)
    img = TrainingImage(
        filename=name,
        path=str(dest),
        n_crops=0,
        content_hash=name,
        duplicate_status=STATUS_UNIQUE if unique else STATUS_EXACT,
        boxes_verified=verified,
        exclude_from_dataset=not unique,
    )
    db.add(img)
    db.flush()
    return img


def _add_crop(
    db,
    dirs,
    img: TrainingImage,
    *,
    grade,
    verified=True,
    skipped=False,
    active=True,
    idx=0,
    pixel_seed: int | None = None,
) -> TrainingLabel:
    crop = dirs["detected"] / str(img.id) / f"seed_{idx}.jpg"
    seed = img.id * 100 + idx if pixel_seed is None else pixel_seed
    _write_jpg(crop, seed=seed)
    lab = TrainingLabel(
        image_id=img.id,
        crop_path=str(crop),
        grade=grade,
        x1=10,
        y1=10,
        x2=50,
        y2=50,
        box_verified=verified,
        active=active,
        skipped=skipped,
        index_in_image=idx,
        crop_hash=None,
    )
    db.add(lab)
    db.flush()
    apply_crop_identity(db, lab)
    img.n_crops = (img.n_crops or 0) + 1
    db.commit()
    return lab


def _five_class_photo(db, dirs, name="all.jpg") -> TrainingImage:
    img = _add_image(db, dirs, name=name)
    for g in range(5):
        _add_crop(db, dirs, img, grade=g, idx=g)
    db.commit()
    return img


def test_inventory_min_crops_gate(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    assert dataset_inventory(db)["dataset_ready"] is False
    img = _add_image(db, dirs)
    for g in range(4):
        _add_crop(db, dirs, img, grade=g, idx=g)
    inv4 = dataset_inventory(db)
    assert inv4["dataset_ready"] is False
    assert inv4["status"] == "NOT READY"
    assert "Current: 4" in (inv4.get("min_crops_message") or "")
    _add_crop(db, dirs, img, grade=4, idx=4)
    inv5 = dataset_inventory(db)
    assert inv5["crops"]["labeled"] == 5
    assert inv5["dataset_ready"] is True
    _add_crop(db, dirs, img, grade=0, idx=5)
    _add_crop(db, dirs, img, grade=1, idx=6)
    inv7 = dataset_inventory(db)
    assert inv7["crops"]["labeled"] == 7
    assert inv7["dataset_ready"] is True


def test_inventory_ready_to_build_when_icdas_3_and_4_missing(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    for idx, g in enumerate([0, 0, 0, 1, 1]):
        _add_crop(db, dirs, img, grade=g, idx=idx)
    inv = dataset_inventory(db)
    assert inv["crops"]["labeled"] == 5
    assert inv["dataset_ready"] is True
    assert inv["status"] == "READY TO BUILD"
    assert inv["classes_ready"] is False
    assert inv["missing_classes"] == [2, 3, 4]
    assert "training is not recommended" in (inv.get("missing_classes_message") or "").lower()


def test_duplicate_unverified_skipped_excluded(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _five_class_photo(db, dirs)
    dup = _add_image(db, dirs, unique=False, name="dup.jpg")
    _add_crop(db, dirs, dup, grade=2, idx=0)
    _add_crop(db, dirs, img, grade=2, idx=20, verified=False)
    _add_crop(db, dirs, img, grade=None, idx=21, skipped=True)
    inv = dataset_inventory(db)
    assert inv["crops"]["labeled"] == 5
    assert inv["photos"]["exact_duplicates"] == 1


def test_label_schema_rejects_icdas_5_and_6():
    for bad in (5, 6, -1):
        with pytest.raises(ValidationError):
            LabelIn(label_id=1, grade=bad)
    assert LabelIn(label_id=1, grade=0).grade == 0
    assert LabelIn(label_id=1, grade=4).grade == 4


def test_save_label_rejects_invalid_and_unverified(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    good = _add_crop(db, dirs, img, grade=None, idx=0)
    with pytest.raises(ValueError):
        save_icdas_label(db, good.id, 5)
    with pytest.raises(ValueError):
        save_icdas_label(db, good.id, 6)
    out = save_icdas_label(db, good.id, 3)
    assert out["grade"] == 3
    unver = _add_crop(db, dirs, img, grade=None, idx=1, verified=False)
    with pytest.raises(ValueError, match="verified"):
        save_icdas_label(db, unver.id, 1)
    img2 = _add_image(db, dirs, verified=False, name="nv.jpg")
    ghost = _add_crop(db, dirs, img2, grade=None, idx=0, verified=False)
    ordered = labeling_queue(db)
    current = ordered.get("current") or {}
    assert current.get("label_id") != ghost.id


def test_invalid_box_rejected_and_edit_delete_add(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs, verified=False)
    lab = _add_crop(db, dirs, img, grade=None, idx=0, verified=False)
    old_hash = lab.crop_hash
    old_path = Path(lab.crop_path)
    old_bytes = old_path.read_bytes()
    with pytest.raises(ValueError, match="Invalid bounding box"):
        save_training_boxes(db, img.id, [{"label_id": lab.id, "x1": 40, "y1": 10, "x2": 10, "y2": 50}], [])
    save_training_boxes(
        db,
        img.id,
        [{"label_id": lab.id, "x1": 12, "y1": 12, "x2": 70, "y2": 72}],
        [],
    )
    db.refresh(lab)
    assert lab.crop_hash != old_hash
    assert Path(lab.crop_path).read_bytes() != old_bytes
    assert "crop_" in Path(lab.crop_path).name
    added = save_training_boxes(
        db,
        img.id,
        [
            {"label_id": lab.id, "x1": 12, "y1": 12, "x2": 70, "y2": 72},
            {"x1": 20, "y1": 20, "x2": 80, "y2": 80},
        ],
        [],
    )
    assert len(added["boxes"]) == 2
    keep_id = added["boxes"][0]["label_id"]
    drop_id = added["boxes"][1]["label_id"]
    save_training_boxes(
        db,
        img.id,
        [{"label_id": keep_id, "x1": 12, "y1": 12, "x2": 70, "y2": 72}],
        [drop_id],
    )
    gone = db.query(TrainingLabel).filter(TrainingLabel.id == drop_id).one()
    assert gone.active is False
    q = labeling_queue(db)
    assert q.get("current") is None or q["current"]["label_id"] != drop_id


def test_validate_xyxy_rejects_nan_and_inverted():
    with pytest.raises(ValueError):
        validate_xyxy_box(float("nan"), 1, 20, 20, 80, 80)
    with pytest.raises(ValueError):
        validate_xyxy_box(30, 1, 10, 20, 80, 80, confidence=0.4)
    with pytest.raises(ValueError):
        validate_xyxy_box(1, 1, 20, 20, 80, 80, confidence=1.5)
    box = validate_xyxy_box(-4, -4, 50, 50, 80, 80, confidence=0.8)
    assert box[0] >= 0 and box[1] >= 0
    assert box_iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_build_dataset_min_split_version_no_train(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _five_class_photo(db, dirs, name="a.jpg")
    img_b = _add_image(db, dirs, name="b.jpg")
    for g in range(5):
        _add_crop(db, dirs, img_b, grade=g, idx=g)
    jobs_before = db.query(TrainingJob).count()
    v1 = build_dataset_version(db)
    assert v1["name"] == "v1"
    assert Path(v1["path"]).exists()
    assert (Path(v1["path"]) / "manifest.csv").exists()
    assert db.query(TrainingJob).count() == jobs_before
    with pytest.raises(ValueError, match="duplicate version"):
        build_dataset_version(db)
    _add_crop(db, dirs, img, grade=0, idx=99)
    v2 = build_dataset_version(db)
    assert v2["name"] == "v2"
    assert Path(v1["path"]).exists()
    assert Path(v2["path"]).exists()
    assert Path(v1["path"]) != Path(v2["path"])
    stats = v2["statistics"]
    assert stats["split_seed"] == 42
    leaks = (stats.get("validation") or {}).get("issues") or []
    assert not any("leaked" in str(x) for x in leaks)
    splits = split_source_images([img.id, img_b.id], seed=42)
    assert not (set(splits["train"]) & set(splits["val"]))
    assert db.query(DatasetVersion).count() == 2


def test_build_rejects_below_minimum(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    for g in range(4):
        _add_crop(db, dirs, img, grade=g, idx=g)
    with pytest.raises(ValueError, match="Need at least 5"):
        build_dataset_version(db)


def test_inventory_ready_to_build_when_some_icdas_classes_missing(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    for i, grade in enumerate((0, 0, 0, 1, 1)):
        _add_crop(db, dirs, img, grade=grade, idx=i)
    inv = dataset_inventory(db)
    assert inv["crops"]["labeled"] == 5
    assert inv["dataset_ready"] is True
    assert inv["status"] == "READY TO BUILD"
    assert inv["classes_ready"] is False
    assert inv["missing_classes"] == [2, 3, 4]
    msg = (inv.get("missing_classes_message") or "").lower()
    assert "2" in msg and "3" in msg and "4" in msg
    assert "can be built" in msg
    assert "training is not recommended" in msg


def test_build_allows_missing_classes_train_gate_does_not(monkeypatch, tmp_path):
    dirs = _patch_wf(monkeypatch, tmp_path)
    db = _session(tmp_path)
    img = _add_image(db, dirs)
    for i, grade in enumerate((0, 0, 0, 1, 1)):
        _add_crop(db, dirs, img, grade=grade, idx=i)
    jobs_before = db.query(TrainingJob).count()
    out = build_dataset_version(db)
    assert out["name"] == "v1"
    assert Path(out["path"]).exists()
    assert db.query(TrainingJob).count() == jobs_before
    assert db.query(DatasetVersion).count() == 1
    with pytest.raises(ValueError, match="All ICDAS classes 0–4 must exist"):
        validate_dataset_for_train(db, None)
