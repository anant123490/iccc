# Lightweight pipeline test. Creates a temporary synthetic fixture only.
# Does not touch a real public dataset.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from common import ensure_dataset_class_dirs, ensure_pipeline_dirs  # noqa: E402


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(">", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)
    return result


def write_jpeg(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    encoded.tofile(str(path))


def make_fixture(root: Path) -> None:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("tooth\ncaries\n", encoding="utf-8")
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(40, 200, size=(320, 480, 3), dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (160, 180), (20, 80, 200), -1)
        cv2.rectangle(img, (220, 90), (360, 230), (200, 40, 40), -1)
        write_jpeg(images / f"mouth_{i:02d}.jpg", img)
        (labels / f"mouth_{i:02d}.txt").write_text(
            "0 0.208 0.344 0.250 0.438\n1 0.604 0.500 0.292 0.438\n",
            encoding="utf-8",
        )
    # Tiny / invalid box should be rejected
    img = rng.integers(40, 200, size=(320, 480, 3), dtype=np.uint8)
    write_jpeg(images / "mouth_bad.jpg", img)
    (labels / "mouth_bad.txt").write_text("0 0.01 0.01 0.01 0.01\n", encoding="utf-8")


def main() -> int:
    ensure_pipeline_dirs()
    ensure_dataset_class_dirs()

    run([sys.executable, str(TOOLS_DIR / "crop_teeth.py"), "--help"])
    run([sys.executable, str(TOOLS_DIR / "build_dataset.py"), "--help"])
    run([sys.executable, str(TOOLS_DIR / "check_dataset.py"), "--help"])

    fixture = ROOT / "tools" / "_tmp_fixture"
    out = ROOT / "tools" / "_tmp_crops"
    if fixture.exists():
        import shutil

        shutil.rmtree(fixture)
    if out.exists():
        import shutil

        shutil.rmtree(out)
    make_fixture(fixture)

    run(
        [
            sys.executable,
            str(TOOLS_DIR / "crop_teeth.py"),
            "--input",
            str(fixture),
            "--output",
            str(out),
            "--format",
            "yolo",
            "--overwrite",
        ]
    )

    crops_csv = out / "crops.csv"
    df = pd.read_csv(crops_csv)
    expected_cols = [
        "crop_id",
        "filename",
        "source_image",
        "annotation_id",
        "annotation_class",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]
    assert list(df.columns)[:11] == expected_cols, list(df.columns)
    assert "icdas_grade" not in df.columns
    assert len(df) >= 8, len(df)
    assert (df["annotation_class"].isin(["tooth", "caries", "0", "1"])).all()

    labels_path = ROOT / "tools" / "_tmp_labels.csv"
    rows = []
    sources = df.groupby("source_image")
    grade = 0
    for source, group in sources:
        assigned = grade % 5
        grade += 1
        for _, rec in group.iterrows():
            rows.append(
                {
                    "crop_id": rec["crop_id"],
                    "filename": rec["filename"],
                    "source_image": rec["source_image"],
                    "icdas_grade": assigned,
                }
            )
    pd.DataFrame(rows).to_csv(labels_path, index=False)

    dataset_tmp = ROOT / "tools" / "_tmp_dataset"
    if dataset_tmp.exists():
        import shutil

        shutil.rmtree(dataset_tmp)

    run(
        [
            sys.executable,
            str(TOOLS_DIR / "build_dataset.py"),
            "--labels",
            str(labels_path),
            "--crops-dir",
            str(out / "images"),
            "--crops-csv",
            str(crops_csv),
            "--dataset",
            str(dataset_tmp),
            "--overwrite",
        ]
    )

    for split in ("train", "val", "test"):
        for cls in ("0", "1", "2", "3", "4"):
            assert (dataset_tmp / split / cls).is_dir(), f"missing {split}/{cls}"

    manifest = pd.read_csv(ROOT / "reports" / "split_manifest.csv")
    leakage = manifest.groupby("source_image")["split"].nunique()
    leaked = leakage[leakage > 1]
    assert leaked.empty, leaked.to_dict()

    run(
        [
            sys.executable,
            str(TOOLS_DIR / "check_dataset.py"),
            "--dataset",
            str(dataset_tmp),
            "--labels",
            str(labels_path),
        ]
    )

    import ast

    ast.parse((TOOLS_DIR / "label_icdas.py").read_text(encoding="utf-8"))
    print("selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
