"""Training dataset helpers (no YOLO, no ICDAS training)."""

from __future__ import annotations

from app.image_phash import average_hash_hex, hamming_hex, is_likely_duplicate
from app.training_workflow import split_source_images


def test_average_hash_identical_images_match():
    import numpy as np

    a = np.full((32, 32, 3), 80, dtype=np.uint8)
    b = a.copy()
    ha = average_hash_hex(a)
    hb = average_hash_hex(b)
    assert ha == hb
    assert hamming_hex(ha, hb) == 0
    assert is_likely_duplicate(ha, hb)


def test_average_hash_flags_near_duplicates():
    import numpy as np

    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[:, :32] = 255
    b = a.copy()
    b[0, 0] = 1
    assert is_likely_duplicate(average_hash_hex(a), average_hash_hex(b))


def test_split_is_by_source_image_without_overlap():
    splits = split_source_images(list(range(1, 21)), seed=42)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == set(range(1, 21))
    n = 20
    assert abs(len(train) / n - 0.70) < 0.08
    again = split_source_images(list(range(1, 21)), seed=42)
    assert again == splits


def test_split_tiny_sets_keep_train():
    s1 = split_source_images([9], seed=42)
    assert s1["train"] == [9]
    s2 = split_source_images([1, 2], seed=42)
    assert len(s2["train"]) == 1
    assert len(s2["val"]) == 1
