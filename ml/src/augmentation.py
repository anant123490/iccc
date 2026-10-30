"""
Training and validation augmentation for ICDAS 0-4.

V6 philosophy
-------------
ICDAS classification depends on subtle visual features such as:

- enamel opacity
- discoloration
- localized lesions
- surface changes
- cavitation
- lesion morphology

Therefore V6 deliberately uses MILD augmentation.

Removed from V6:

- Perspective distortion
- CoarseDropout
- aggressive color changes

These transformations can hide or distort clinically relevant
visual evidence.

Important:
-----------
The current dataset pipeline materializes augmented images when
the TensorFlow dataset is created. Therefore this file only defines
the transformations; dynamic per-epoch augmentation should be
implemented later as a separate controlled experiment.
"""

from __future__ import annotations

import albumentations as A


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

def get_train_augmentation(
    image_size: int = 224,
) -> A.Compose:
    """
    Return mild augmentation for ICDAS 0-4 classification.

    The goal is to improve robustness without destroying
    subtle lesion information.
    """

    return A.Compose(
        [
            # ------------------------------------------------
            # Small geometric variation
            # ------------------------------------------------

            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.08,
                rotate_limit=10,
                border_mode=0,
                p=0.5,
            ),

            # ------------------------------------------------
            # Horizontal orientation variation
            # ------------------------------------------------

            A.HorizontalFlip(
                p=0.5,
            ),

            # ------------------------------------------------
            # Mild illumination variation
            # ------------------------------------------------

            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.30,
            ),

            # ------------------------------------------------
            # Very mild color variation
            # ------------------------------------------------

            A.HueSaturationValue(
                hue_shift_limit=3,
                sat_shift_limit=8,
                val_shift_limit=5,
                p=0.15,
            ),

            # ------------------------------------------------
            # Very mild blur
            # ------------------------------------------------

            A.GaussianBlur(
                blur_limit=(3, 3),
                p=0.05,
            ),

            # ------------------------------------------------
            # Final resize
            # ------------------------------------------------

            A.Resize(
                image_size,
                image_size,
            ),
        ]
    )


# ============================================================
# VALIDATION / TEST AUGMENTATION
# ============================================================

def get_val_augmentation(
    image_size: int = 224,
) -> A.Compose:
    """
    Validation and test preprocessing.

    No random augmentation is applied.
    """

    return A.Compose(
        [
            A.Resize(
                image_size,
                image_size,
            )
        ]
    )