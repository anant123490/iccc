"""Progress rules: no per-photo class forcing."""

from app.admin_progress import (
    MIN_DATASET_CROPS,
    crop_is_resolved,
    dataset_build_ready,
    dataset_classes_ready,
    dataset_min_crops_message,
    imbalance_warning_text,
    missing_classes_message,
    missing_icdas_classes,
    validate_icdas_grade,
)


def test_mixed_grades_on_one_photo_are_valid():
    assert crop_is_resolved(0, False)
    assert crop_is_resolved(1, False)
    assert crop_is_resolved(4, False)
    assert crop_is_resolved(None, True)
    assert not crop_is_resolved(None, False)


def test_dataset_not_ready_if_class_2_or_3_missing():
    counts = {"0": 10, "1": 8, "2": 0, "3": 0, "4": 5}
    assert missing_icdas_classes(counts) == [2, 3]
    assert not dataset_classes_ready(counts)
    msg = missing_classes_message([2, 3])
    assert "2" in msg and "3" in msg
    assert "training is not recommended" in msg.lower()
    assert dataset_build_ready(23, counts) is True


def test_dataset_ready_when_all_classes_present():
    counts = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1}
    assert dataset_classes_ready(counts)
    assert missing_icdas_classes(counts) == []


def test_min_dataset_crops_is_five():
    assert MIN_DATASET_CROPS == 5
    assert dataset_min_crops_message(0) == "Need at least 5 verified labeled UNIQUE crops. Current: 0."
    assert dataset_min_crops_message(4) == "Need at least 5 verified labeled UNIQUE crops. Current: 4."
    assert dataset_min_crops_message(5) is None
    assert dataset_min_crops_message(7) is None
    all_classes = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1}
    assert not dataset_build_ready(0, all_classes)
    assert not dataset_build_ready(4, all_classes)
    assert dataset_build_ready(5, all_classes)
    assert dataset_build_ready(7, all_classes)
    four_classes = {"0": 2, "1": 1, "2": 1, "3": 1, "4": 0}
    assert dataset_build_ready(5, four_classes)
    assert not dataset_classes_ready(four_classes)
    only_01 = {"0": 10, "1": 5, "2": 0, "3": 0, "4": 0}
    assert dataset_build_ready(15, only_01)
    assert not dataset_classes_ready(only_01)


def test_icdas_grade_rejects_five_and_six():
    for g in range(5):
        assert validate_icdas_grade(g) == g
    for bad in (5, 6, -1, None, True, "3", 2.7):
        try:
            validate_icdas_grade(bad)
            raise AssertionError(f"expected reject {bad!r}")
        except ValueError:
            pass


def test_imbalance_does_not_block_labeling():
    text = imbalance_warning_text({"0": 100, "1": 2, "2": 2, "3": 2, "4": 2})
    assert text
    assert "do not assign" in text.lower()
