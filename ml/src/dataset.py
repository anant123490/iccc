"""
ICDAS dataset loading and TensorFlow pipeline.

Supports ICDAS 0–4 only. Directories for other grades (including 5 and 6)
are rejected rather than remapped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from .augmentation import get_train_augmentation, get_val_augmentation
from .icdas import NUM_CLASSES, SPLITS, VALID_CLASS_NAMES
from .preprocessing import preprocess_image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetValidationError(ValueError):
    """Raised when the on-disk dataset does not match ICDAS 0–4."""


def discover_images_from_folders(
    root: str,
    num_classes: int = NUM_CLASSES,
    splits: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """
    Discover images from:

        dataset/
            train|val|test/
                0/ 1/ 2/ 3/ 4/

    (canonical root: data/icdas/)

    Raises DatasetValidationError if any split contains a class directory
    other than 0–4. ICDAS 5/6 images must not be folded into class 4.
    """
    records = []
    root_path = Path(root)
    allowed = {str(i) for i in range(num_classes)}

    selected_splits = SPLITS if splits is None else tuple(splits)
    invalid_splits = sorted(set(selected_splits) - set(SPLITS))
    if invalid_splits:
        raise ValueError(f"Unsupported dataset split(s): {invalid_splits}")

    for split in selected_splits:
        split_dir = root_path / split
        if split == "val" and not split_dir.exists() and (root_path / "validation").exists():
            split_dir = root_path / "validation"
        if not split_dir.exists():
            continue

        labels_path = split_dir / "labels.csv"
        if labels_path.is_file():
            try:
                labels = pd.read_csv(labels_path)
            except Exception as exc:
                raise DatasetValidationError(f"Could not read {labels_path}: {exc}") from exc
            if not {"filename", "icdas_score"}.issubset(labels.columns):
                raise DatasetValidationError(
                    f"{labels_path} must include filename and icdas_score columns."
                )
            for row in labels.to_dict(orient="records"):
                try:
                    label = int(row["icdas_score"])
                except (TypeError, ValueError) as exc:
                    raise DatasetValidationError(
                        f"Invalid ICDAS label {row.get('icdas_score')!r} in {labels_path}."
                    ) from exc
                if label < 0 or label >= num_classes:
                    raise DatasetValidationError(
                        f"Unsupported ICDAS label {label} in {labels_path}; this pipeline supports 0–4 only."
                    )
                filename = str(row.get("filename") or "").strip()
                if not filename:
                    raise DatasetValidationError(f"Empty filename in {labels_path}.")
                image_path = Path(filename)
                if not image_path.is_absolute():
                    image_path = split_dir / image_path
                if not image_path.is_file():
                    raise DatasetValidationError(f"Missing tooth crop referenced by {labels_path}: {image_path}")
                records.append(
                    {
                        "filename": str(image_path.relative_to(root_path))
                        if image_path.is_relative_to(root_path)
                        else str(image_path),
                        "icdas_score": label,
                        "split": split,
                    }
                )
            continue

        extra_dirs = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            name = class_dir.name
            if name not in allowed:
                extra_dirs.append(name)
                continue

            label = int(name)
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                records.append(
                    {
                        "filename": str(image_path.relative_to(root_path)),
                        "icdas_score": label,
                        "split": split,
                    }
                )

        if extra_dirs:
            raise DatasetValidationError(
                f"Unsupported class directories in {split_dir}: {sorted(extra_dirs)}. "
                f"This project supports only ICDAS {', '.join(VALID_CLASS_NAMES)}. "
                "Do not move ICDAS 5 or 6 images into class 4. "
                "Relocate them outside train/val/test (for example data/icdas/excluded/)."
            )

    return pd.DataFrame(records)


def validate_dataset_layout(
    root: str,
    num_classes: int = NUM_CLASSES,
    require_all_splits: bool = True,
    require_all_classes: bool = True,
) -> Dict[str, Dict[int, int]]:
    """
    Validate folder layout and return per-split class counts.

    Raises DatasetValidationError on unsupported directories or missing classes.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise DatasetValidationError(f"Dataset root does not exist: {root_path}")

    allowed = {str(i) for i in range(num_classes)}
    report: Dict[str, Dict[int, int]] = {}

    if any(
        (root_path / folder / "labels.csv").is_file()
        for folder in ("train", "val", "validation", "test")
    ):
        data = discover_images_from_folders(str(root_path), num_classes=num_classes)
        found_splits = set(data["split"].tolist()) if not data.empty else set()
        for split in SPLITS:
            if split not in found_splits:
                if require_all_splits:
                    raise DatasetValidationError(f"No tooth crops found in split '{split}'.")
                continue
            counts = {c: int((data.loc[data["split"] == split, "icdas_score"] == c).sum()) for c in range(num_classes)}
            missing = [c for c, count in counts.items() if count == 0]
            if require_all_classes and missing:
                raise DatasetValidationError(f"Missing images for classes {missing} in split '{split}'.")
            report[split] = counts
        return report

    for split in SPLITS:
        split_dir = root_path / split
        if not split_dir.exists():
            if require_all_splits:
                raise DatasetValidationError(
                    f"Missing required split directory: {split_dir}"
                )
            continue

        extra = [
            p.name
            for p in split_dir.iterdir()
            if p.is_dir() and p.name not in allowed
        ]
        if extra:
            raise DatasetValidationError(
                f"Unsupported class directories in {split}: {sorted(extra)}. "
                f"Allowed: {sorted(allowed)}. "
                "ICDAS 5/6 must not be remapped to ICDAS 4."
            )

        counts = {c: 0 for c in range(num_classes)}
        for class_name in sorted(allowed, key=int):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                if require_all_classes:
                    raise DatasetValidationError(
                        f"Missing class directory {class_dir}"
                    )
                continue
            n = sum(
                1
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            )
            counts[int(class_name)] = n

        missing = [c for c, n in counts.items() if n == 0]
        if require_all_classes and missing:
            raise DatasetValidationError(
                f"Missing images for classes {missing} in split '{split}'."
            )
        report[split] = counts

    return report


def format_dataset_report(report: Dict[str, Dict[int, int]]) -> str:
    lines = []
    for split, counts in report.items():
        lines.append(split.upper())
        for grade, n in counts.items():
            lines.append(f"{grade}: {n}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class DentalCariesDataset:
    """TensorFlow dataset wrapper for ICDAS 0–4."""

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 224,
        batch_size: int = 16,
        augment: bool = False,
        preprocess_cfg: Optional[Dict] = None,
        num_classes: int = NUM_CLASSES,
    ):
        if augment and split != "train":
            raise ValueError(
                "Augmentation is only allowed on the training split "
                "to avoid validation/test leakage."
            )

        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        self.preprocess_cfg = preprocess_cfg or {}
        self.num_classes = num_classes

        self.aug = (
            get_train_augmentation(image_size)
            if augment
            else get_val_augmentation(image_size)
        )

        all_data = discover_images_from_folders(
            str(self.root),
            num_classes=num_classes,
            splits=(split,),
        )
        if len(all_data) == 0:
            raise FileNotFoundError(f"No images found inside {self.root}")

        self.df = all_data[all_data["split"] == split].copy()
        self.df.reset_index(drop=True, inplace=True)
        if len(self.df) == 0:
            raise ValueError(f"No images found for split '{split}'.")

    def validate_classes(self, num_classes: int | None = None):
        num_classes = self.num_classes if num_classes is None else num_classes
        counts = self.df["icdas_score"].value_counts()
        invalid = sorted(
            int(c) for c in counts.index if int(c) < 0 or int(c) >= num_classes
        )
        if invalid:
            raise DatasetValidationError(
                f"Found unsupported ICDAS labels {invalid} in {self.split}. "
                "Classes 5 and 6 must not be used or remapped to 4."
            )
        missing = [c for c in range(num_classes) if counts.get(c, 0) == 0]
        if missing:
            raise DatasetValidationError(
                f"Missing classes in {self.split}: {missing}\n"
                f"Class counts:\n{counts.sort_index()}"
            )

    def class_distribution(self) -> Dict[int, int]:
        counts = self.df["icdas_score"].value_counts()
        return {int(c): int(counts.get(c, 0)) for c in sorted(counts.index)}

    def _load_sample(self, row) -> Tuple[np.ndarray, int]:
        path = Path(row["filename"])
        if not path.is_absolute():
            path = self.root / path

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        image = preprocess_image(
            image,
            target_size=self.image_size,
            **self.preprocess_cfg,
        )
        # preprocess_image returns RGB float32 [0, 255].
        image = np.clip(image, 0.0, 255.0)
        image_uint8 = image.astype(np.uint8)

        if self.augment:
            image_uint8 = self.aug(image=image_uint8)["image"]

        image = np.clip(
            image_uint8.astype(np.float32),
            0.0,
            255.0,
        )
        label = int(row["icdas_score"])
        if label < 0 or label >= self.num_classes:
            raise DatasetValidationError(
                f"Label {label} is outside ICDAS 0–{self.num_classes - 1}."
            )
        return image, label

    def as_tf_dataset(self, shuffle: bool = True) -> tf.data.Dataset:
        images = []
        labels = []
        for _, row in self.df.iterrows():
            try:
                image, label = self._load_sample(row)
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"WARNING: skipping {row['filename']}: {e}")

        if len(images) == 0:
            raise RuntimeError(f"No valid images found in {self.split}")

        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(labels),
                seed=42,
                reshuffle_each_iteration=True,
            )
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def get_class_weights(self, num_classes: int | None = None) -> Dict[int, float]:
        num_classes = self.num_classes if num_classes is None else num_classes
        counts = self.df["icdas_score"].value_counts()
        total = len(self.df)
        weights = {}
        for c in range(num_classes):
            count = int(counts.get(c, 0))
            if count == 0:
                raise ValueError(
                    f"Cannot calculate class weight: class {c} has zero images."
                )
            weights[c] = total / (num_classes * count)
        return weights

    def print_distribution(self, num_classes: int | None = None):
        num_classes = self.num_classes if num_classes is None else num_classes
        counts = self.df["icdas_score"].value_counts()
        print(f"\n{self.split.upper()} DATASET")
        total = len(self.df)
        for c in range(num_classes):
            count = int(counts.get(c, 0))
            percentage = count / total * 100 if total > 0 else 0
            print(f"Grade {c}: {count:4d} ({percentage:5.1f}%)")
        print(f"Total: {total}\n")
