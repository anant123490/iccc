from __future__ import annotations

from types import SimpleNamespace

from app.admin_portal_ops import resolve_image_splits


def _image(image_id, patient_id, **overrides):
    values = {
        "id": image_id,
        "patient_id": patient_id,
        "content_hash": f"hash-{image_id}",
        "path": "",
        "split_name": "UNASSIGNED",
        "split_locked": False,
        "canonical_id": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_percentage_assignment_keeps_three_patient_groups_nonempty():
    images = [_image(1, "PAT001"), _image(2, "PAT002"), _image(3, "PAT003")]
    result = resolve_image_splits(images, {}, {"strategy": "percentage", "seed": 42})

    assert set(result["image_to_split"]) == {1, 2, 3}
    assert {result["image_to_split"][image_id] for image_id in result["image_to_split"]} == {
        "train", "val", "test"
    }


def test_exact_sha256_fallback_keeps_duplicate_patients_together(tmp_path):
    first = tmp_path / "first.jpg"
    duplicate = tmp_path / "duplicate.jpg"
    first.write_bytes(b"same exact photo bytes")
    duplicate.write_bytes(b"same exact photo bytes")
    images = [
        _image(1, "PAT001", content_hash=None, path=str(first)),
        _image(2, "PAT002", content_hash=None, path=str(duplicate)),
        _image(3, "PAT003"),
        _image(4, "PAT004"),
    ]

    result = resolve_image_splits(images, {}, {"strategy": "percentage", "seed": 11})

    assert result["image_to_split"][1] == result["image_to_split"][2]
    assert not result["patient_conflicts"]


def test_manual_patient_assignment_applies_to_every_photo():
    images = [
        _image(1, "PAT001"),
        _image(2, "PAT001"),
        _image(3, "PAT002"),
        _image(4, "PAT003"),
    ]
    result = resolve_image_splits(
        images,
        {},
        {
            "strategy": "manual",
            "assignments": {"1": "train", "2": "train", "3": "val", "4": "test"},
        },
    )

    assert result["image_to_split"] == {1: "train", 2: "train", 3: "val", 4: "test"}
    assert result["issues"] == []


def test_exact_photo_counts_reject_patient_group_splitting():
    images = [
        _image(1, "PAT001"),
        _image(2, "PAT001"),
        _image(3, "PAT002"),
        _image(4, "PAT002"),
        _image(5, "PAT003"),
    ]
    result = resolve_image_splits(
        images,
        {},
        {"strategy": "counts", "train_images": 3, "val_images": 1, "test_images": 1},
    )

    assert any("without splitting a patient group" in issue for issue in result["issues"])
    assert not result["image_to_split"]


def test_missing_patient_id_stays_unassigned_with_warning():
    images = [
        _image(1, None),
        _image(2, "PAT001"),
        _image(3, "PAT002"),
        _image(4, "PAT003"),
    ]
    result = resolve_image_splits(images, {}, {"strategy": "percentage"})

    assert 1 in result["unassigned_image_ids"]
    assert len(result["missing_patient_ids"]) == 1
    assert result["warnings"]
    assert not result["issues"]
