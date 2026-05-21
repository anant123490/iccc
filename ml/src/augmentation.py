"""
Data augmentation simulating real smartphone capture conditions.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2  # noqa: F401 — optional PyTorch bridge


def get_train_augmentation(image_size: int = 224) -> A.Compose:
    """Training augmentation pipeline."""
    return A.Compose(
        [
            A.Rotate(limit=15, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.8
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                p=0.4,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.4),
            A.HorizontalFlip(p=0.5),
            A.Resize(image_size, image_size),
        ]
    )


def get_val_augmentation(image_size: int = 224) -> A.Compose:
    """Validation/test — resize only."""
    return A.Compose([A.Resize(image_size, image_size)])
