#!/usr/bin/env python3
"""
Stratified 70/15/15 split for ICDAS 0–4 images.

Moves original files only. Augmentation is never applied here, so
validation/test cannot contain augmented copies of training images.

ICDAS 5 and 6 folders are moved to dataset/excluded/ and are never
relabeled as class 4.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.icdas import NUM_CLASSES, SPLITS, VALID_CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_class_files(root: Path, grade: int) -> list[Path]:
    files: list[Path] = []
    for split in SPLITS:
        files.extend(list_images(root / split / str(grade)))
    return files


def split_files(files: list[Path], seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    if n == 0:
        return [], [], []
    if n == 1:
        return files, [], []
    if n == 2:
        return [files[0]], [files[1]], []

    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    n_train = n - n_test - n_val
    train = files[:n_train]
    val = files[n_train : n_train + n_val]
    test = files[n_train + n_val :]
    return train, val, test


def exclude_unsupported(root: Path) -> None:
    excluded = root / "excluded"
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for extra in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if extra.name in VALID_CLASS_NAMES:
                continue
            dest = excluded / extra.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in list_images(extra):
                target = dest / img.name
                if target.exists():
                    target = dest / f"{split}_{img.name}"
                shutil.move(str(img), str(target))
                print(f"Excluded ICDAS {extra.name}: {img.name} -> {target}")
            extra.rmdir()
            print(
                f"Removed unsupported class directory {extra} "
                "(not remapped to ICDAS 4)."
            )


def ensure_class_dirs(root: Path) -> None:
    for split in SPLITS:
        for grade in range(NUM_CLASSES):
            (root / split / str(grade)).mkdir(parents=True, exist_ok=True)


def unique_dest(dest_dir: Path, name: str) -> Path:
    candidate = dest_dir / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "data" / "icdas"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    exclude_unsupported(root)
    ensure_class_dirs(root)

    staging = root / "_split_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    for grade in range(NUM_CLASSES):
        files = collect_class_files(root, grade)
        train, val, test = split_files(files, seed=args.seed + grade)
        print(f"ICDAS {grade}: train={len(train)} val={len(val)} test={len(test)}")
        mapping = {"train": train, "val": val, "test": test}
        for split, split_files_ in mapping.items():
            dest_dir = staging / split / str(grade)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files_:
                shutil.move(str(src), str(unique_dest(dest_dir, src.name)))

    for split in SPLITS:
        for grade in range(NUM_CLASSES):
            live = root / split / str(grade)
            live.mkdir(parents=True, exist_ok=True)
            for leftover in list_images(live):
                leftover.unlink()
            staged = staging / split / str(grade)
            if staged.exists():
                for img in list_images(staged):
                    shutil.move(str(img), str(live / img.name))

    shutil.rmtree(staging)
    print("Split complete. Augmentation is applied only during training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
