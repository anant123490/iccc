"""
Data augmentation for ICDAS training.
"""

import albumentations as A


def get_train_augmentation(
    image_size: int = 224,
) -> A.Compose:

    return A.Compose(
        [

            A.Rotate(
                limit=15,
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),

            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.15,
            ),

            A.HorizontalFlip(
                p=0.5,
            ),

            A.Perspective(
                scale=(0.02, 0.05),
                p=0.2,
            ),

            A.Resize(
                image_size,
                image_size,
            ),
        ]
    )


def get_val_augmentation(
    image_size: int = 224,
) -> A.Compose:

    return A.Compose(
        [
            A.Resize(
                image_size,
                image_size,
            )
        ]
    )