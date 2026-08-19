"""
ICDAS dataset loading and TensorFlow pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from .preprocessing import preprocess_image
from .augmentation import get_train_augmentation, get_val_augmentation


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images_from_folders(root: str) -> pd.DataFrame:
    """
    Discover images from:

        dataset/
            train/
                0/
                1/
                ...
                6/
            val/
                0/
                ...
                6/
            test/
                0/
                ...
                6/

    Returns:
        DataFrame with filename, icdas_score and split.
    """

    records = []
    root_path = Path(root)

    for split in ["train", "val", "test"]:

        split_dir = root_path / split

        if not split_dir.exists():
            continue

        for class_dir in sorted(split_dir.iterdir()):

            if not class_dir.is_dir():
                continue

            if not class_dir.name.isdigit():
                continue

            label = int(class_dir.name)

            if label < 0 or label > 6:
                raise ValueError(
                    f"Invalid ICDAS class {label} in {class_dir}"
                )

            for image_path in class_dir.iterdir():

                if image_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                records.append(
                    {
                        "filename": str(
                            image_path.relative_to(root_path)
                        ),
                        "icdas_score": label,
                        "split": split,
                    }
                )

    return pd.DataFrame(records)


class DentalCariesDataset:
    """TensorFlow dataset wrapper for ICDAS."""

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 224,
        batch_size: int = 16,
        augment: bool = False,
        preprocess_cfg: Optional[Dict] = None,
    ):

        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        self.preprocess_cfg = preprocess_cfg or {}

        if augment:
            self.aug = get_train_augmentation(image_size)
        else:
            self.aug = get_val_augmentation(image_size)

        all_data = discover_images_from_folders(str(self.root))

        if len(all_data) == 0:
            raise FileNotFoundError(
                f"No images found inside {self.root}"
            )

        self.df = all_data[
            all_data["split"] == split
        ].copy()

        self.df.reset_index(drop=True, inplace=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No images found for split '{split}'."
            )

    def validate_classes(self, num_classes: int = 7):
        """
        Ensure every ICDAS class exists.
        """

        counts = self.df["icdas_score"].value_counts()

        missing = [
            c for c in range(num_classes)
            if counts.get(c, 0) == 0
        ]

        if missing:
            raise ValueError(
                f"Missing classes in {self.split}: {missing}\n"
                f"Class counts:\n{counts.sort_index()}"
            )

    def class_distribution(self) -> Dict[int, int]:
        """
        Return class counts.
        """

        counts = self.df["icdas_score"].value_counts()

        return {
            int(c): int(counts.get(c, 0))
            for c in sorted(counts.index)
        }

    def _load_sample(
        self,
        row,
    ) -> Tuple[np.ndarray, int]:

        path = Path(row["filename"])

        if not path.is_absolute():
            path = self.root / path

        image = cv2.imread(str(path))

        if image is None:
            raise ValueError(
                f"Failed to load image: {path}"
            )

        # Existing preprocessing pipeline
        image = preprocess_image(
            image,
            target_size=self.image_size,
            **self.preprocess_cfg,
        )

        # preprocess_image is expected to return [0,1]
        image = np.clip(image, 0.0, 1.0)

        # Albumentations expects uint8
        image_uint8 = (
            image * 255.0
        ).astype(np.uint8)

        if self.augment:
            augmented = self.aug(
                image=image_uint8
            )

            image_uint8 = augmented["image"]

        image = (
            image_uint8.astype(np.float32)
            / 255.0
        )

        label = int(row["icdas_score"])

        return image, label

    def as_tf_dataset(
        self,
        shuffle: bool = True,
    ) -> tf.data.Dataset:

        images = []
        labels = []

        for _, row in self.df.iterrows():

            try:

                image, label = self._load_sample(row)

                images.append(image)
                labels.append(label)

            except Exception as e:

                print(
                    f"WARNING: skipping "
                    f"{row['filename']}: {e}"
                )

        if len(images) == 0:
            raise RuntimeError(
                f"No valid images found in {self.split}"
            )

        images = np.asarray(
            images,
            dtype=np.float32
        )

        labels = np.asarray(
            labels,
            dtype=np.int32
        )

        ds = tf.data.Dataset.from_tensor_slices(
            (images, labels)
        )

        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(labels),
                seed=42,
                reshuffle_each_iteration=True,
            )

        ds = ds.batch(
            self.batch_size
        )

        ds = ds.prefetch(
            tf.data.AUTOTUNE
        )

        return ds

    def get_class_weights(
        self,
        num_classes: int,
    ) -> Dict[int, float]:

        counts = self.df[
            "icdas_score"
        ].value_counts()

        total = len(self.df)

        weights = {}

        for c in range(num_classes):

            count = int(counts.get(c, 0))

            if count == 0:
                raise ValueError(
                    f"Cannot calculate class weight: "
                    f"class {c} has zero images."
                )

            weights[c] = (
                total
                / (num_classes * count)
            )

        return weights

    def print_distribution(
        self,
        num_classes: int = 7,
    ):

        counts = self.df[
            "icdas_score"
        ].value_counts()

        print(
            f"\n{self.split.upper()} DATASET"
        )

        total = len(self.df)

        for c in range(num_classes):

            count = int(counts.get(c, 0))

            percentage = (
                count / total * 100
                if total > 0
                else 0
            )

            print(
                f"Grade {c}: "
                f"{count:4d} "
                f"({percentage:5.1f}%)"
            )

        print(f"Total: {total}\n")