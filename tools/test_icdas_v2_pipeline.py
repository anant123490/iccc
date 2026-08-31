"""Unit tests for ICDAS v2 label rules (no d/D mapping, no 5/6, SKIP excluded)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from icdas_v2_lib import (  # noqa: E402
    group_key,
    is_valid_grade,
    persist_label,
    stable_split,
)


def test_grades():
    assert is_valid_grade("0")
    assert is_valid_grade("SKIP")
    assert not is_valid_grade("5")
    assert not is_valid_grade("D")
    assert not is_valid_grade("d")


def test_persist_and_skip(tmp_path, monkeypatch):
    import icdas_v2_lib as lib

    monkeypatch.setattr(lib, "DATA_ICDAS", tmp_path)

    def lp():
        return tmp_path / "manifest" / "icdas_labels.csv"

    monkeypatch.setattr(lib, "labels_path", lp)
    lib.ensure_icdas_dirs()
    persist_label(
        {
            "sample_id": "s1",
            "source_type": "user",
            "source_image": "a.jpg",
            "crop_path": "c.jpg",
            "icdas_grade": "2",
        }
    )
    persist_label(
        {
            "sample_id": "s2",
            "source_type": "public",
            "source_image": "b.jpg",
            "crop_path": "d.jpg",
            "icdas_grade": "SKIP",
        }
    )
    with pytest.raises(ValueError):
        persist_label({"sample_id": "s3", "icdas_grade": "5"})
    with pytest.raises(ValueError):
        persist_label({"sample_id": "s4", "icdas_grade": "D"})
    df = pd.read_csv(tmp_path / "manifest" / "icdas_labels.csv", dtype=str)
    assert set(df["icdas_grade"]) == {"2", "SKIP"}
    train = df[(df["status"] == "labelled") & (df["icdas_grade"].isin(list("01234")))]
    assert list(train["sample_id"]) == ["s1"]


def test_same_source_same_split():
    r1 = pd.Series({"source_image": "mouth.jpg", "sample_id": "a"})
    r2 = pd.Series({"source_image": "mouth.jpg", "sample_id": "b"})
    k1, k2 = group_key(r1), group_key(r2)
    assert k1 == k2
    assert stable_split(k1, 42, 0.7, 0.15) == stable_split(k2, 42, 0.7, 0.15)
