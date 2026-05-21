"""
Dataset loading with CSV annotation support and TensorFlow data pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from .preprocessing import preprocess_image
from .augmentation import get_train_augmentation, get_val_augmentation


def load_annotations(csv_path: str, split: Optional[str] = None) -> pd.DataFrame:
    """
    Load annotations.csv with columns: filename, icdas_score, split (optional)
    """
    df = pd.read_csv(csv_path)
    required = {"filename", "icdas_score"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    if split and "split" in df.columns:
        df = df[df["split"] == split]
    return df


def discover_images_from_folders(root: str) -> pd.DataFrame:
    """
    Build annotation DataFrame from folder structure:
    dataset/train/0/, dataset/train/1/, ... or flat with labels in filename
    """
    records = []
    for split in ["train", "val", "test"]:
        split_dir = Path(root) / split
        if not split_dir.exists():
            continue
        # Class subfolders
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name.isdigit():
                label = int(class_dir.name)
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        records.append(
                            {
                                "filename": str(img_path),
                                "icdas_score": label,
                                "split": split,
                            }
                        )
    return pd.DataFrame(records)


class DentalCariesDataset:
    """TensorFlow dataset wrapper for ICDAS training."""

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 224,
        batch_size: int = 32,
        augment: bool = False,
        preprocess_cfg: Optional[Dict] = None,
        annotations_file: str = "annotations.csv",
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        self.preprocess_cfg = preprocess_cfg or {}
        self.aug = get_train_augmentation(image_size) if augment else get_val_augmentation(image_size)

        csv_path = self.root / annotations_file
        folder_df = discover_images_from_folders(str(self.root))
        if split and len(folder_df):
            folder_df = folder_df[folder_df["split"] == split]

        if csv_path.exists():
            csv_df = load_annotations(str(csv_path), split=split)
            # Prefer folders when CSV is stale (e.g. new images copied without updating CSV)
            if len(folder_df) > len(csv_df):
                print(
                    f"Warning: {len(folder_df)} images in folders vs {len(csv_df)} in "
                    f"{annotations_file} for split '{split}'. Using folder paths. "
                    "Run: python ml/scripts/sync_annotations.py"
                )
                self.df = folder_df
            else:
                self.df = csv_df
        else:
            self.df = folder_df

        if len(self.df) == 0:
            raise FileNotFoundError(
                f"No images found for split '{split}' in {root}. "
                "Run: python ml/scripts/setup_dataset.py"
            )

    def _load_sample(self, row) -> Tuple[np.ndarray, int]:
        path = row["filename"]
        if not os.path.isabs(path):
            rel = Path(path)
            if rel.parts and rel.parts[0] in ("train", "val", "test"):
                path = str(self.root / rel)
            else:
                path = str(self.root / self.split / rel)
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load: {path}")
        image = (preprocess_image(image, target_size=self.image_size, **self.preprocess_cfg) * 255).astype(
            np.uint8
        )
        if self.augment:
            augmented = self.aug(image=image)
            image = augmented["image"]
        label = int(row["icdas_score"])
        return image.astype(np.float32) / 255.0, label

    def as_tf_dataset(self, shuffle: bool = True) -> tf.data.Dataset:
        """Convert to tf.data.Dataset."""
        images, labels = [], []
        for _, row in self.df.iterrows():
            try:
                img, lbl = self._load_sample(row)
                images.append(img)
                labels.append(lbl)
            except Exception as e:
                print(f"Warning: skip {row.get('filename')}: {e}")

        ds = tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))
        if shuffle:
            ds = ds.shuffle(min(len(labels), 1000), seed=42)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_class_weights(self, num_classes: int) -> Dict[int, float]:
        """Compute inverse frequency class weights."""
        counts = self.df["icdas_score"].value_counts()
        total = len(self.df)
        weights = {}
        for c in range(num_classes):
            n = counts.get(c, 1)
            weights[c] = total / (num_classes * n)
        return weights
