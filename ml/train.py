#!/usr/bin/env python3

"""
ICDAS 0-4 Training Pipeline — V6 CLEAN

Purpose
-------
Controlled baseline experiment for ICDAS 0-4 classification.

V6 changes
----------
1. Standard SparseCategoricalCrossentropy.
2. No Keras class_weight.
3. No custom class weights.
4. Mild augmentation is controlled by dataset.py.
5. Images remain in [0,255] before MobileNetV3 preprocessing.
6. QWK is used for checkpoint selection.
7. Two-stage training is retained.
8. ICDAS 5 and 6 remain completely out of scope.
9. Nested v6_clean.yaml configuration is supported correctly.
10. Dataset path is read from dataset.root.
11. Output path is read from output.directory.
12. Model/training settings are read from their nested sections.

IMPORTANT
---------
Do NOT modify the v77 test set during training/tuning.

Expected dataset:

data/training/versions/v77/
    train/
        0/
        1/
        2/
        3/
        4/

    val/
        0/
        1/
        2/
        3/
        4/

    test/
        0/
        1/
        2/
        3/
        4/

ICDAS 5 and 6 are NOT supported.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras


# ============================================================
# PATH SETUP
# ============================================================

ML_DIR = Path(__file__).resolve().parent

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.dataset import DentalCariesDataset

from src.icdas import NUM_CLASSES

from src.losses import (
    ordinal_loss,
    ordinal_predict,
    ordinal_to_class_probabilities,
    QuadraticWeightedKappa,
)

from src.metrics import save_evaluation_report

from src.model import (
    build_model,
    get_custom_objects,
    unfreeze_top_layers,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("icdas.train")


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_NUM_CLASSES = 5

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# CONFIG LOADER
# ============================================================

def load_config(path: str) -> dict:
    """
    Load YAML configuration.

    The V6 YAML uses nested sections such as:

        dataset:
            root: ...

        model:
            dropout: ...

        stage1:
            epochs: ...

    """
    config_path = Path(path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"YAML configuration is empty: {config_path}"
        )

    if not isinstance(config, dict):
        raise ValueError(
            "YAML root must be a dictionary."
        )

    logger.info(
        "Configuration loaded: %s",
        config_path,
    )

    return config


# ============================================================
# CONFIG HELPERS
# ============================================================

def get_nested(
    config: dict,
    section: str,
    key: str,
    default=None,
):
    """
    Read a value from nested YAML configuration.

    Example:

        dataset:
            root: data/training/versions/v77

    get_nested(config, "dataset", "root")
    -> data/training/versions/v77

    Legacy flat configuration is also supported.
    """

    section_data = config.get(section)

    if isinstance(section_data, dict):
        if key in section_data:
            return section_data[key]

    # Backward compatibility with older flat YAML files.
    flat_key = f"{section}_{key}"

    if flat_key in config:
        return config[flat_key]

    return default


def get_dataset_root(
    config: dict,
    project_root: Path,
) -> Path:
    """
    Resolve dataset root.

    Priority:
        1. dataset.root
        2. dataset_root
        3. data/icdas
    """

    dataset_root_value = get_nested(
        config,
        "dataset",
        "root",
        None,
    )

    if dataset_root_value is None:
        dataset_root_value = config.get(
            "dataset_root"
        )

    if dataset_root_value is None:
        dataset_root_value = "data/icdas"

    dataset_root = Path(
        str(dataset_root_value)
    )

    if not dataset_root.is_absolute():
        dataset_root = (
            project_root / dataset_root
        )

    return dataset_root.resolve()


def get_output_directory(
    config: dict,
    project_root: Path,
) -> Path:
    """
    Resolve output directory.

    Priority:
        1. output.directory
        2. output_dir
        3. models/icdas/current/<experiment_name>
    """

    output_value = get_nested(
        config,
        "output",
        "directory",
        None,
    )

    if output_value is None:
        output_value = config.get(
            "output_dir"
        )

    if output_value is not None:
        output_dir = Path(
            str(output_value)
        )

        if not output_dir.is_absolute():
            output_dir = (
                project_root / output_dir
            )

        return output_dir.resolve()

    experiment_name = config.get(
        "experiment_name",
        "icdas_mobilenet_cbam_5class_v6_clean",
    )

    return (
        project_root
        / "models"
        / "icdas"
        / "current"
        / str(experiment_name)
    ).resolve()


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config(
    config: dict,
):
    """
    Validate V6 configuration.

    Supports the nested v6_clean.yaml structure.
    """

    dataset_cfg = config.get(
        "dataset",
        {},
    )

    model_cfg = config.get(
        "model",
        {},
    )

    loss_cfg = config.get(
        "loss",
        {},
    )

    stage1_cfg = config.get(
        "stage1",
        {},
    )

    stage2_cfg = config.get(
        "stage2",
        {},
    )

    # --------------------------------------------------------
    # Number of classes
    # --------------------------------------------------------

    num_classes = int(
        dataset_cfg.get(
            "num_classes",
            model_cfg.get(
                "num_classes",
                config.get(
                    "num_classes",
                    EXPECTED_NUM_CLASSES,
                ),
            ),
        )
    )

    if num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            "This project supports ICDAS 0-4 only. "
            "num_classes must be 5."
        )

    # --------------------------------------------------------
    # Image size
    # --------------------------------------------------------

    image_size = int(
        dataset_cfg.get(
            "image_size",
            model_cfg.get(
                "input_shape",
                [224, 224, 3],
            )[0]
            if isinstance(
                model_cfg.get(
                    "input_shape"
                ),
                list,
            )
            else config.get(
                "image_size",
                224,
            ),
        )
    )

    if image_size != 224:
        logger.warning(
            "Current image_size=%s. "
            "Recommended value is 224.",
            image_size,
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    if isinstance(
        loss_cfg,
        dict,
    ):
        loss_name = str(
            loss_cfg.get(
                "name",
                "sparse_categorical_crossentropy",
            )
        ).lower()
    else:
        loss_name = str(
            loss_cfg
        ).lower()

    # --------------------------------------------------------
    # Ordinal
    # --------------------------------------------------------

    ordinal_regression = bool(
        model_cfg.get(
            "ordinal_regression",
            config.get(
                "ordinal_regression",
                False,
            ),
        )
    )

    # --------------------------------------------------------
    # Log configuration
    # --------------------------------------------------------

    logger.info(
        "Configuration validated."
    )

    logger.info(
        "Classes: %d",
        num_classes,
    )

    logger.info(
        "Image size: %d",
        image_size,
    )

    logger.info(
        "Loss: %s",
        loss_cfg,
    )

    logger.info(
        "Ordinal regression: %s",
        ordinal_regression,
    )

    logger.info(
        "Stage 1 enabled: %s",
        stage1_cfg.get(
            "enabled",
            True,
        ),
    )

    logger.info(
        "Stage 1 epochs: %s",
        stage1_cfg.get(
            "epochs",
            20,
        ),
    )

    logger.info(
        "Stage 2 enabled: %s",
        stage2_cfg.get(
            "enabled",
            True,
        ),
    )

    logger.info(
        "Stage 2 epochs: %s",
        stage2_cfg.get(
            "epochs",
            30,
        ),
    )


# ============================================================
# DATASET INPUT SCALE
# ============================================================

def convert_dataset_to_mobilenet_range(
    dataset: tf.data.Dataset,
) -> tf.data.Dataset:
    """
    Ensure images are float32 [0,255].

    dataset.py currently returns [0,255].

    This function protects against older [0,1]
    datasets without double scaling.
    """

    def transform(
        images,
        labels,
    ):
        images = tf.cast(
            images,
            tf.float32,
        )

        max_value = tf.reduce_max(
            images
        )

        def from_unit_interval():
            return (
                tf.clip_by_value(
                    images,
                    0.0,
                    1.0,
                )
                * 255.0
            )

        def already_byte_range():
            return tf.clip_by_value(
                images,
                0.0,
                255.0,
            )

        images = tf.cond(
            max_value <= 1.5,
            from_unit_interval,
            already_byte_range,
        )

        return (
            images,
            labels,
        )

    return dataset.map(
        transform,
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# ============================================================
# DATASET SANITY CHECK
# ============================================================

def inspect_dataset(
    dataset: tf.data.Dataset,
    name: str,
):
    """
    Inspect one batch.

    Checks:
        shape
        min
        max
        mean
        std
        labels
    """

    try:
        images, labels = next(
            iter(dataset)
        )
    except StopIteration:
        raise RuntimeError(
            f"{name} dataset is empty."
        )

    images_np = images.numpy()

    logger.info(
        "----------------------------------------"
    )

    logger.info(
        "%s DATASET SANITY CHECK",
        name,
    )

    logger.info(
        "Shape: %s",
        images_np.shape,
    )

    logger.info(
        "Min: %.4f",
        images_np.min(),
    )

    logger.info(
        "Max: %.4f",
        images_np.max(),
    )

    logger.info(
        "Mean: %.4f",
        images_np.mean(),
    )

    logger.info(
        "Std: %.4f",
        images_np.std(),
    )

    logger.info(
        "Labels: %s",
        labels.numpy(),
    )

    logger.info(
        "----------------------------------------"
    )

    if images_np.max() <= 1.1:

        logger.warning(
            "%s images are approximately "
            "in [0,1].",
            name,
        )

    elif images_np.max() > 255.5:

        logger.warning(
            "%s images contain values above 255.",
            name,
        )

    else:

        logger.info(
            "%s input scale is correct: [0,255].",
            name,
        )


# ============================================================
# MODEL COMPILATION
# ============================================================

def compile_model(
    model,
    config: dict,
    learning_rate: float,
):
    """
    Compile V6 model.

    V6 baseline intentionally uses:

        SparseCategoricalCrossentropy

    and does NOT use:

        class weights
        Keras class_weight
        focal loss

    Ordinal loss remains available for future experiments,
    but is disabled in v6_clean.yaml.
    """

    model_cfg = config.get(
        "model",
        {},
    )

    loss_cfg = config.get(
        "loss",
        {},
    )

    ordinal_regression = bool(
        model_cfg.get(
            "ordinal_regression",
            config.get(
                "ordinal_regression",
                False,
            ),
        )
    )

    if isinstance(
        loss_cfg,
        dict,
    ):

        loss_name = str(
            loss_cfg.get(
                "name",
                "sparse_categorical_crossentropy",
            )
        ).lower()

    else:

        loss_name = str(
            loss_cfg
        ).lower()

    num_classes = int(
        config.get(
            "dataset",
            {},
        ).get(
            "num_classes",
            model_cfg.get(
                "num_classes",
                EXPECTED_NUM_CLASSES,
            ),
        )
    )

    # ========================================================
    # ORDINAL
    # ========================================================

    if (
        ordinal_regression
        or loss_name == "ordinal"
    ):

        loss = ordinal_loss(
            num_classes
        )

        logger.info(
            "Using ordinal regression loss."
        )

    # ========================================================
    # FOCAL
    # ========================================================

    elif loss_name == "focal":

        from src.losses import focal_loss

        gamma = 2.0
        alpha = 0.25

        if isinstance(
            loss_cfg,
            dict,
        ):

            gamma = float(
                loss_cfg.get(
                    "gamma",
                    config.get(
                        "focal_gamma",
                        2.0,
                    ),
                )
            )

            alpha = float(
                loss_cfg.get(
                    "alpha",
                    config.get(
                        "focal_alpha",
                        0.25,
                    ),
                )
            )

        loss = focal_loss(
            gamma=gamma,
            alpha=alpha,
        )

        logger.info(
            "Using focal loss."
        )

    # ========================================================
    # STANDARD CE
    # ========================================================

    else:

        label_smoothing = 0.0

        if isinstance(
            loss_cfg,
            dict,
        ):

            label_smoothing = float(
                loss_cfg.get(
                    "label_smoothing",
                    0.0,
                )
            )

        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )

        logger.info(
            "Using SparseCategoricalCrossentropy."
        )

        logger.info(
            "Label smoothing: %.4f",
            label_smoothing,
        )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer_cfg = config.get(
        "optimizer",
        {},
    )

    weight_decay = float(
        optimizer_cfg.get(
            "weight_decay",
            config.get(
                "weight_decay",
                1e-4,
            ),
        )
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=float(
            learning_rate
        ),
        weight_decay=weight_decay,
    )

    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            QuadraticWeightedKappa(
                num_classes=num_classes
            )
        ],
    )

    logger.info(
        "Learning rate: %.8f",
        learning_rate,
    )

    logger.info(
        "Weight decay: %.8f",
        weight_decay,
    )

    return model


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(
    output_dir: Path,
    config: dict,
):
    """
    Create callbacks.

    QWK is the model-selection metric.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "best.keras"
    )

    callbacks_cfg = config.get(
        "callbacks",
        {},
    )

    early_cfg = callbacks_cfg.get(
        "early_stopping",
        {},
    )

    reduce_cfg = callbacks_cfg.get(
        "reduce_lr",
        {},
    )

    checkpoint_cfg = callbacks_cfg.get(
        "checkpoint",
        {},
    )

    patience = int(
        early_cfg.get(
            "patience",
            8,
        )
    )

    lr_patience = int(
        reduce_cfg.get(
            "patience",
            3,
        )
    )

    monitor = str(
        callbacks_cfg.get(
            "monitor",
            "val_qwk",
        )
    )

    mode = str(
        callbacks_cfg.get(
            "mode",
            "max",
        )
    )

    callback_list = []

    # ========================================================
    # CHECKPOINT
    # ========================================================

    if checkpoint_cfg.get(
        "enabled",
        True,
    ):

        callback_list.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(
                    checkpoint_path
                ),
                monitor=monitor,
                mode=mode,
                save_best_only=bool(
                    checkpoint_cfg.get(
                        "save_best_only",
                        True,
                    )
                ),
                verbose=1,
            )
        )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if early_cfg.get(
        "enabled",
        True,
    ):

        callback_list.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=patience,
                restore_best_weights=bool(
                    early_cfg.get(
                        "restore_best_weights",
                        True,
                    )
                ),
                verbose=1,
            )
        )

    # ========================================================
    # REDUCE LR
    # ========================================================

    if reduce_cfg.get(
        "enabled",
        True,
    ):

        callback_list.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=str(
                    reduce_cfg.get(
                        "monitor",
                        "val_loss",
                    )
                ),
                mode="min",
                factor=float(
                    reduce_cfg.get(
                        "factor",
                        0.5,
                    )
                ),
                patience=lr_patience,
                min_lr=float(
                    reduce_cfg.get(
                        "min_lr",
                        1e-7,
                    )
                ),
                verbose=1,
            )
        )

    # ========================================================
    # HISTORY
    # ========================================================

    callback_list.append(
        keras.callbacks.CSVLogger(
            str(
                output_dir
                / "history.csv"
            )
        )
    )

    return callback_list


# ============================================================
# PREDICTION DECODER
# ============================================================

def decode_predictions(
    raw,
    ordinal_regression: bool,
):
    """
    Convert raw model output into:

        predictions
        probabilities
    """

    raw = np.asarray(
        raw
    )

    if ordinal_regression:

        predictions = (
            ordinal_predict(
                tf.convert_to_tensor(
                    raw
                )
            )
            .numpy()
        )

        probabilities = (
            ordinal_to_class_probabilities(
                raw
            )
        )

        return (
            predictions,
            probabilities,
        )

    predictions = np.argmax(
        raw,
        axis=-1,
    )

    return (
        predictions,
        raw,
    )


# ============================================================
# PREDICTION SANITY CHECK
# ============================================================

def prediction_sanity_check(
    model,
    dataset,
    ordinal_regression=False,
):
    """
    Run prediction on one batch and print
    prediction distribution.
    """

    images, labels = next(
        iter(dataset)
    )

    raw = model.predict(
        images,
        verbose=0,
    )

    predictions, probabilities = (
        decode_predictions(
            raw,
            ordinal_regression,
        )
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "MODEL PREDICTION SANITY CHECK"
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "TRUE LABELS: %s",
        labels.numpy(),
    )

    logger.info(
        "PREDICTIONS: %s",
        predictions,
    )

    logger.info(
        "PROBABILITIES:"
    )

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    print(
        probabilities
    )

    unique, counts = np.unique(
        predictions,
        return_counts=True,
    )

    logger.info(
        "Prediction distribution: %s",
        dict(
            zip(
                unique.tolist(),
                counts.tolist(),
            )
        ),
    )

    if len(unique) == 1:

        logger.warning(
            "WARNING: model predicts only "
            "one class in this batch."
        )

    logger.info(
        "========================================"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataset,
    num_classes: int,
    ordinal_regression: bool = False,
    output_dir: Path | None = None,
):
    """
    Evaluate model on the complete test dataset.
    """

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in dataset:

        raw_output = model.predict(
            images,
            verbose=0,
        )

        predictions, probabilities = (
            decode_predictions(
                raw_output,
                ordinal_regression,
            )
        )

        y_true.extend(
            labels.numpy().tolist()
        )

        y_pred.extend(
            predictions.tolist()
        )

        y_prob.extend(
            probabilities.tolist()
        )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    y_prob = np.asarray(
        y_prob
    )

    if len(y_true) == 0:
        raise RuntimeError(
            "Test dataset produced zero samples."
        )

    # ========================================================
    # METRICS
    # ========================================================

    if output_dir is not None:

        evaluation_dir = (
            Path(output_dir)
            / "test_evaluation"
        )

        metrics = save_evaluation_report(
            y_true,
            y_pred,
            y_prob,
            num_classes,
            str(evaluation_dir),
        )

    else:

        from src.metrics import compute_metrics

        metrics = compute_metrics(
            y_true,
            y_pred,
            num_classes,
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"TEST ACCURACY : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"MACRO F1      : "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"WEIGHTED F1   : "
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )

    print(
        "========================================"
    )

    print(
        "\nPER-CLASS RESULTS:"
    )

    for grade, row in metrics[
        "per_class"
    ].items():

        print(
            f"ICDAS {grade}: "
            f"precision={row['precision'] * 100:.1f}% "
            f"recall={row['recall'] * 100:.1f}% "
            f"f1={row['f1'] * 100:.1f}% "
            f"(n={row['support']})"
        )

    print(
        "\nCONFUSION MATRIX:"
    )

    print(
        np.array(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    print(
        "\nCLASSIFICATION REPORT:"
    )

    print(
        metrics[
            "classification_report"
        ]
    )

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: dict,
    output_dir: Path,
):
    """
    Save final test metrics as JSON.
    """

    path = (
        output_dir
        / "test_results.json"
    )

    serializable = {}

    for key, value in results.items():

        if key == "classification_report":
            continue

        if isinstance(
            value,
            np.ndarray,
        ):
            serializable[key] = (
                value.tolist()
            )

        else:
            serializable[key] = value

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            serializable,
            f,
            indent=2,
        )

        f.write("\n")

    logger.info(
        "Saved results: %s",
        path,
    )


# ============================================================
# DATASET CREATION
# ============================================================

def create_datasets(
    dataset_root: Path,
    config: dict,
    num_classes: int,
):
    """
    Create train, validation and test datasets.

    Uses:

        train
        val
        test

    from v77.

    The 'valid' directory is intentionally NOT used.
    """

    dataset_cfg = config.get(
        "dataset",
        {},
    )

    preprocessing_cfg = config.get(
        "preprocessing",
        {},
    )

    augmentation_cfg = config.get(
        "augmentation",
        {},
    )

    image_size = int(
        dataset_cfg.get(
            "image_size",
            224,
        )
    )

    batch_size = int(
        dataset_cfg.get(
            "batch_size",
            8,
        )
    )

    train_split = str(
        dataset_cfg.get(
            "train_split",
            "train",
        )
    )

    val_split = str(
        dataset_cfg.get(
            "val_split",
            "val",
        )
    )

    test_split = str(
        dataset_cfg.get(
            "test_split",
            "test",
        )
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    preprocess_cfg = {
        "use_roi": bool(
            preprocessing_cfg.get(
                "use_roi_detection",
                False,
            )
        ),

        "use_clahe": bool(
            preprocessing_cfg.get(
                "use_clahe",
                False,
            )
        ),

        "use_specular": bool(
            preprocessing_cfg.get(
                "use_specular_reduction",
                False,
            )
        ),

        "color_norm": bool(
            preprocessing_cfg.get(
                "color_normalize",
                False,
            )
        ),
    }

    logger.info(
        "Preprocessing configuration: %s",
        preprocess_cfg,
    )

    # ========================================================
    # AUGMENTATION
    # ========================================================

    augmentation_enabled = bool(
        augmentation_cfg.get(
            "enabled",
            True,
        )
    )

    logger.info(
        "Training augmentation enabled: %s",
        augmentation_enabled,
    )

    # ========================================================
    # DATASET ROOT CHECK
    # ========================================================

    logger.info(
        "Checking dataset root: %s",
        dataset_root,
    )

    if not dataset_root.exists():

        raise FileNotFoundError(
            "Dataset root does not exist:\n"
            f"{dataset_root}\n\n"
            "Expected V77 dataset at:\n"
            "data/training/versions/v77"
        )

    # ========================================================
    # TRAIN
    # ========================================================

    train_data = DentalCariesDataset(
        str(dataset_root),
        train_split,
        image_size=image_size,
        batch_size=batch_size,
        augment=augmentation_enabled,
        preprocess_cfg=preprocess_cfg,
        num_classes=num_classes,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_data = DentalCariesDataset(
        str(dataset_root),
        val_split,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        preprocess_cfg=preprocess_cfg,
        num_classes=num_classes,
    )

    # ========================================================
    # TEST
    # ========================================================

    test_data = DentalCariesDataset(
        str(dataset_root),
        test_split,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        preprocess_cfg=preprocess_cfg,
        num_classes=num_classes,
    )

    # ========================================================
    # VALIDATE CLASSES
    # ========================================================

    train_data.validate_classes(
        num_classes
    )

    val_data.validate_classes(
        num_classes
    )

    test_data.validate_classes(
        num_classes
    )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    train_data.print_distribution(
        num_classes
    )

    val_data.print_distribution(
        num_classes
    )

    test_data.print_distribution(
        num_classes
    )

    # ========================================================
    # TF DATASETS
    # ========================================================

    train_ds = train_data.as_tf_dataset(
        shuffle=True
    )

    val_ds = val_data.as_tf_dataset(
        shuffle=False
    )

    test_ds = test_data.as_tf_dataset(
        shuffle=False
    )

    # ========================================================
    # INPUT SCALE
    # ========================================================

    logger.info(
        "Ensuring dataset uses MobileNetV3 "
        "input range [0,255]."
    )

    train_ds = (
        convert_dataset_to_mobilenet_range(
            train_ds
        )
    )

    val_ds = (
        convert_dataset_to_mobilenet_range(
            val_ds
        )
    )

    test_ds = (
        convert_dataset_to_mobilenet_range(
            test_ds
        )
    )

    # ========================================================
    # PREFETCH
    # ========================================================

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        tf.data.AUTOTUNE
    )

    test_ds = test_ds.prefetch(
        tf.data.AUTOTUNE
    )

    return (
        train_ds,
        val_ds,
        test_ds,
    )


# ============================================================
# BUILD MODEL
# ============================================================

def create_model(
    config: dict,
    num_classes: int,
):
    """
    Build MobileNetV3Small + CBAM.
    """

    model_cfg = config.get(
        "model",
        {},
    )

    image_size = int(
        config.get(
            "dataset",
            {},
        ).get(
            "image_size",
            224,
        )
    )

    dropout = float(
        model_cfg.get(
            "dropout",
            0.30,
        )
    )

    attention_type = model_cfg.get(
        "attention_type",
        model_cfg.get(
            "attention",
            config.get(
                "use_attention",
                "cbam",
            ),
        ),
    )

    pretrained = bool(
        model_cfg.get(
            "pretrained",
            True,
        )
    )

    ordinal_regression = bool(
        model_cfg.get(
            "ordinal_regression",
            False,
        )
    )

    logger.info(
        "Building model:"
    )

    logger.info(
        "  Backbone: MobileNetV3Small"
    )

    logger.info(
        "  Attention: %s",
        attention_type,
    )

    logger.info(
        "  Dropout: %.2f",
        dropout,
    )

    logger.info(
        "  Pretrained: %s",
        pretrained,
    )

    logger.info(
        "  Ordinal regression: %s",
        ordinal_regression,
    )

    model = build_model(
        num_classes=num_classes,
        image_size=image_size,
        attention_type=attention_type,
        dropout=dropout,
        pretrained=pretrained,
        ordinal_regression=ordinal_regression,
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "Train ICDAS 0-4 classifier — V6 CLEAN."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "ml/configs/default.yaml"
        ),
        help="Path to YAML configuration.",
    )

    args = parser.parse_args()

    # ========================================================
    # CONFIG
    # ========================================================

    config = load_config(
        args.config
    )

    validate_config(
        config
    )

    # ========================================================
    # PROJECT ROOT
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    # ========================================================
    # DATASET ROOT
    # ========================================================

    dataset_root = get_dataset_root(
        config,
        project_root,
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = get_output_directory(
        config,
        project_root,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # BASIC CONFIG VALUES
    # ========================================================

    dataset_cfg = config.get(
        "dataset",
        {},
    )

    model_cfg = config.get(
        "model",
        {},
    )

    stage1_cfg = config.get(
        "stage1",
        {},
    )

    stage2_cfg = config.get(
        "stage2",
        {},
    )

    num_classes = int(
        dataset_cfg.get(
            "num_classes",
            model_cfg.get(
                "num_classes",
                EXPECTED_NUM_CLASSES,
            ),
        )
    )

    ordinal_regression = bool(
        model_cfg.get(
            "ordinal_regression",
            False,
        )
    )

    # ========================================================
    # SEED
    # ========================================================

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    np.random.seed(
        seed
    )

    tf.random.set_seed(
        seed
    )

    keras.utils.set_random_seed(
        seed
    )

    # ========================================================
    # HEADER
    # ========================================================

    logger.info(
        "========================================"
    )

    logger.info(
        "ICDAS 0-4 TRAINING — V6 CLEAN"
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "Project root: %s",
        project_root,
    )

    logger.info(
        "Dataset root: %s",
        dataset_root,
    )

    logger.info(
        "Output directory: %s",
        output_dir,
    )

    logger.info(
        "Classes: ICDAS 0-4"
    )

    logger.info(
        "Loss: SparseCategoricalCrossentropy"
    )

    logger.info(
        "Class weighting: DISABLED"
    )

    logger.info(
        "Keras class_weight: DISABLED"
    )

    logger.info(
        "Focal loss: DISABLED"
    )

    logger.info(
        "Ordinal regression: %s",
        ordinal_regression,
    )

    logger.info(
        "Dataset configuration: %s",
        dataset_cfg,
    )

    logger.info(
        "========================================"
    )

    # ========================================================
    # SAFETY CHECK — DATASET
    # ========================================================

    if not dataset_root.exists():

        raise FileNotFoundError(
            "\nV77 DATASET NOT FOUND.\n\n"
            f"Expected:\n{dataset_root}\n\n"
            "Your YAML should contain:\n"
            "dataset:\n"
            '  root: "data/training/versions/v77"\n'
        )

    logger.info(
        "V77 dataset root exists."
    )

    # ========================================================
    # DATASETS
    # ========================================================

    (
        train_ds,
        val_ds,
        test_ds,
    ) = create_datasets(
        dataset_root,
        config,
        num_classes,
    )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    inspect_dataset(
        train_ds,
        "TRAIN",
    )

    inspect_dataset(
        val_ds,
        "VALIDATION",
    )

    inspect_dataset(
        test_ds,
        "TEST",
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = create_model(
        config,
        num_classes,
    )

    model.summary()

    # ========================================================
    # STAGE 1
    # ========================================================

    stage1_enabled = bool(
        stage1_cfg.get(
            "enabled",
            True,
        )
    )

    if stage1_enabled:

        logger.info(
            "========================================"
        )

        logger.info(
            "STAGE 1"
        )

        logger.info(
            "Training classification head."
        )

        logger.info(
            "========================================"
        )

        # ----------------------------------------------------
        # COMPILE STAGE 1
        # ----------------------------------------------------

        stage1_lr = float(
            stage1_cfg.get(
                "learning_rate",
                config.get(
                    "optimizer",
                    {},
                ).get(
                    "learning_rate",
                    1e-4,
                ),
            )
        )

        model = compile_model(
            model,
            config,
            learning_rate=stage1_lr,
        )

        # ----------------------------------------------------
        # EPOCHS
        # ----------------------------------------------------

        stage1_epochs = int(
            stage1_cfg.get(
                "epochs",
                20,
            )
        )

        logger.info(
            "Stage 1 epochs: %d",
            stage1_epochs,
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage1_epochs,
            callbacks=get_callbacks(
                output_dir,
                config,
            ),
            verbose=1,
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        best_path = (
            output_dir
            / "best.keras"
        )

        if not best_path.exists():

            raise FileNotFoundError(
                "Best Stage 1 model was not created:\n"
                f"{best_path}"
            )

        logger.info(
            "Best Stage 1 checkpoint: %s",
            best_path,
        )

        # ----------------------------------------------------
        # LOAD BEST STAGE 1
        # ----------------------------------------------------

        model = keras.models.load_model(
            str(best_path),
            compile=False,
            custom_objects=(
                get_custom_objects()
            ),
        )

        # ----------------------------------------------------
        # STAGE 1 SANITY
        # ----------------------------------------------------

        logger.info(
            "Stage 1 prediction sanity check."
        )

        prediction_sanity_check(
            model,
            val_ds,
            ordinal_regression,
        )

    else:

        logger.info(
            "Stage 1 disabled."
        )

        best_path = (
            output_dir
            / "best.keras"
        )

    # ========================================================
    # STAGE 2
    # ========================================================

    stage2_enabled = bool(
        stage2_cfg.get(
            "enabled",
            True,
        )
    )

    if stage2_enabled:

        logger.info(
            "========================================"
        )

        logger.info(
            "STAGE 2"
        )

        logger.info(
            "Fine-tuning top MobileNetV3 layers."
        )

        logger.info(
            "========================================"
        )

        # ----------------------------------------------------
        # UNFREEZE
        # ----------------------------------------------------

        unfreeze_layers = int(
            stage2_cfg.get(
                "unfreeze_layers",
                20,
            )
        )

        logger.info(
            "Requested MobileNet layers to unfreeze: %d",
            unfreeze_layers,
        )

        unfreeze_top_layers(
            model,
            num_layers=unfreeze_layers,
        )

        # ----------------------------------------------------
        # RECOMPILE
        # ----------------------------------------------------

        stage2_lr = float(
            stage2_cfg.get(
                "learning_rate",
                1e-5,
            )
        )

        model = compile_model(
            model,
            config,
            learning_rate=stage2_lr,
        )

        # ----------------------------------------------------
        # EPOCHS
        # ----------------------------------------------------

        stage2_epochs = int(
            stage2_cfg.get(
                "epochs",
                30,
            )
        )

        logger.info(
            "Stage 2 epochs: %d",
            stage2_epochs,
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage2_epochs,
            callbacks=get_callbacks(
                output_dir,
                config,
            ),
            verbose=1,
        )

    else:

        logger.info(
            "Stage 2 disabled."
        )

    # ========================================================
    # FINAL BEST CHECKPOINT
    # ========================================================

    best_path = (
        output_dir
        / "best.keras"
    )

    if not best_path.exists():

        raise FileNotFoundError(
            "Final best model was not created:\n"
            f"{best_path}"
        )

    # ========================================================
    # LOAD FINAL BEST
    # ========================================================

    model = keras.models.load_model(
        str(best_path),
        compile=False,
        custom_objects=(
            get_custom_objects()
        ),
    )

    # ========================================================
    # FINAL SANITY
    # ========================================================

    logger.info(
        "Running final prediction sanity check."
    )

    prediction_sanity_check(
        model,
        test_ds,
        ordinal_regression,
    )

    # ========================================================
    # SAVE DEPLOYMENT MODEL
    # ========================================================

    deploy_path = (
        output_dir
        / "final.keras"
    )

    model.save(
        str(deploy_path)
    )

    # ========================================================
    # OPTIONAL ROOT CHECKPOINTS
    # ========================================================

    overwrite_root_checkpoints = bool(
        config.get(
            "overwrite_root_checkpoints",
            False,
        )
    )

    if overwrite_root_checkpoints:

        models_root = (
            project_root
            / "models"
            / "icdas"
            / "current"
        )

        root_deploy = (
            models_root
            / "deploy.keras"
        )

        root_best_path = (
            models_root
            / "best.keras"
        )

        shutil.copy2(
            deploy_path,
            root_deploy,
        )

        shutil.copy2(
            best_path,
            root_best_path,
        )

        logger.info(
            "Root deployment checkpoint updated."
        )

    # ========================================================
    # MODEL SAVED
    # ========================================================

    logger.info(
        "========================================"
    )

    logger.info(
        "MODEL SAVED"
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "Best: %s",
        best_path,
    )

    logger.info(
        "Deployment: %s",
        deploy_path,
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    logger.info(
        "========================================"
    )

    logger.info(
        "FINAL TEST EVALUATION"
    )

    logger.info(
        "========================================"
    )

    results = evaluate_model(
        model,
        test_ds,
        num_classes,
        ordinal_regression=ordinal_regression,
        output_dir=output_dir,
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    save_results(
        results,
        output_dir,
    )

    # ========================================================
    # FINISHED
    # ========================================================

    logger.info(
        "========================================"
    )

    logger.info(
        "V6 CLEAN TRAINING COMPLETE"
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "Best model: %s",
        best_path,
    )

    logger.info(
        "Deployment model: %s",
        deploy_path,
    )

    logger.info(
        "Test results: %s",
        output_dir
        / "test_results.json",
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()