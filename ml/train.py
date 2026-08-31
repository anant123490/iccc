#!/usr/bin/env python3
"""
ICDAS 0-4 Training Pipeline

Features:
    - 5 ICDAS classes: 0, 1, 2, 3, 4
    - MobileNetV3-Small
    - CBAM attention
    - ImageNet pretrained weights
    - Training augmentation
    - Class-weighted training
    - Optional focal loss
    - Optional ordinal regression
    - Two-stage training
    - Early stopping
    - ReduceLROnPlateau
    - Best-model checkpointing
    - Full test evaluation
    - Confusion matrix
    - Classification report

IMPORTANT
---------
DentalCariesDataset currently returns images normalized to [0, 1].

Keras MobileNetV3 with ImageNet pretrained weights and built-in
preprocessing expects images in the [0, 255] range.

Therefore this training pipeline converts:

    [0, 1] -> [0, 255]

before sending images into MobileNetV3.

This same preprocessing contract MUST also be used during inference.
"""


from __future__ import annotations

import argparse
import json
import logging
import os
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
)

from src.metrics import (
    save_evaluation_report,
)

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
# YAML LOADER
# ============================================================

def load_config(path: str) -> dict:
    """
    Load YAML configuration.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"YAML configuration is empty: {path}"
        )

    return config


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config(config: dict):
    """
    Validate important training configuration.
    """

    num_classes = int(
        config.get(
            "num_classes",
            EXPECTED_NUM_CLASSES,
        )
    )

    if num_classes != EXPECTED_NUM_CLASSES:

        raise ValueError(
            f"This project is configured for ICDAS 0-4 "
            f"only. Expected num_classes=5, "
            f"but got {num_classes}."
        )

    image_size = int(
        config.get(
            "image_size",
            224,
        )
    )

    if image_size != 224:

        logger.warning(
            "MobileNetV3 training is normally configured "
            "for 224x224. Current image_size=%s",
            image_size,
        )

    loss_name = str(
        config.get(
            "loss",
            "sparse_categorical_crossentropy",
        )
    ).lower()

    ordinal_regression = bool(
        config.get(
            "ordinal_regression",
            False,
        )
    )

    if ordinal_regression and loss_name not in {
        "ordinal",
        "sparse_categorical_crossentropy",
        "focal",
    }:

        raise ValueError(
            f"Invalid loss configuration: {loss_name}"
        )

    logger.info(
        "Configuration validated successfully."
    )

    logger.info(
        "Number of classes: %d",
        num_classes,
    )

    logger.info(
        "Image size: %d",
        image_size,
    )

    logger.info(
        "Loss: %s",
        loss_name,
    )

    logger.info(
        "Ordinal regression: %s",
        ordinal_regression,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights_from_dataset(
    dataset_root: Path,
    num_classes: int,
):
    """
    Calculate balanced class weights.

    Formula:

        weight =
            total_samples /
            (num_classes * class_count)
    """

    train_root = (
        dataset_root
        / "train"
    )

    if not train_root.exists():

        raise FileNotFoundError(
            f"Training directory not found: "
            f"{train_root}"
        )

    counts = {}

    logger.info(
        "Calculating class weights from: %s",
        train_root,
    )

    for class_id in range(num_classes):

        class_dir = (
            train_root
            / str(class_id)
        )

        if not class_dir.exists():

            raise FileNotFoundError(
                f"Missing training class directory: "
                f"{class_dir}"
            )

        count = sum(
            1
            for file in class_dir.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in VALID_EXTENSIONS
            )
        )

        if count == 0:

            raise ValueError(
                f"Class {class_id} contains "
                f"zero training images."
            )

        counts[class_id] = count

    total_samples = sum(
        counts.values()
    )

    class_weights = {}

    for class_id in range(num_classes):

        class_weights[class_id] = (
            total_samples
            / (
                num_classes
                * counts[class_id]
            )
        )

    logger.info(
        "========================================"
    )

    logger.info(
        "TRAINING CLASS DISTRIBUTION"
    )

    logger.info(
        "========================================"
    )

    for class_id in range(num_classes):

        logger.info(
            "ICDAS %d: %d images -> weight %.4f",
            class_id,
            counts[class_id],
            class_weights[class_id],
        )

    logger.info(
        "========================================"
    )

    return class_weights


# ============================================================
# DATASET INPUT SCALE FIX
# ============================================================

def convert_dataset_to_mobilenet_range(
    dataset: tf.data.Dataset,
) -> tf.data.Dataset:
    """
    Ensure dataset images are float32 [0, 255] for MobileNetV3.

    DentalCariesDataset now returns [0, 255]. Older [0, 1]
    batches are scaled once. Values already in [0, 255]
    are not multiplied again.
    """

    def transform(
        images,
        labels,
    ):

        images = tf.cast(
            images,
            tf.float32,
        )

        max_value = tf.reduce_max(images)

        def from_unit_interval():
            return (
                tf.clip_by_value(images, 0.0, 1.0)
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
    Print image statistics.

    This is extremely important for detecting
    preprocessing mismatch.
    """

    images, labels = next(
        iter(dataset)
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
            "%s images are still approximately "
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
            "%s input scale looks correct "
            "for MobileNetV3 [0,255].",
            name,
        )


# ============================================================
# MODEL COMPILATION
# ============================================================

def compile_model(
    model,
    config,
    learning_rate: float,
):
    """
    Compile model according to YAML.
    """

    ordinal_regression = bool(
        config.get(
            "ordinal_regression",
            False,
        )
    )

    loss_name = str(
        config.get(
            "loss",
            "sparse_categorical_crossentropy",
        )
    ).lower()

    num_classes = int(
        config.get(
            "num_classes",
            EXPECTED_NUM_CLASSES,
        )
    )

    # ========================================================
    # ORDINAL LOSS
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
    # FOCAL LOSS
    # ========================================================

    elif loss_name == "focal":

        from src.losses import focal_loss

        loss = focal_loss(
            gamma=float(
                config.get(
                    "focal_gamma",
                    2.0,
                )
            ),
            alpha=float(
                config.get(
                    "focal_alpha",
                    0.25,
                )
            ),
        )

        logger.info(
            "Using focal loss."
        )

    # ========================================================
    # STANDARD CLASSIFICATION
    # ========================================================

    else:

        loss = (
            keras.losses
            .SparseCategoricalCrossentropy()
        )

        logger.info(
            "Using SparseCategoricalCrossentropy."
        )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = (
        keras.optimizers.AdamW(
            learning_rate=float(
                learning_rate
            ),
            weight_decay=float(
                config.get(
                    "weight_decay",
                    1e-4,
                )
            ),
        )
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    logger.info(
        "Learning rate: %.8f",
        learning_rate,
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
    Training callbacks.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "best.keras"
    )

    patience = int(
        config.get(
            "early_stopping_patience",
            10,
        )
    )

    lr_patience = int(
        config.get(
            "reduce_lr_patience",
            3,
        )
    )

    return [

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        keras.callbacks.ModelCheckpoint(

            filepath=str(
                checkpoint_path
            ),

            monitor="val_loss",

            mode="min",

            save_best_only=True,

            verbose=1,
        ),

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        keras.callbacks.EarlyStopping(

            monitor="val_loss",

            mode="min",

            patience=patience,

            restore_best_weights=True,

            verbose=1,
        ),

        # ----------------------------------------------------
        # LEARNING RATE REDUCTION
        # ----------------------------------------------------

        keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            mode="min",

            factor=0.3,

            patience=lr_patience,

            min_lr=1e-7,

            verbose=1,
        ),

        # ----------------------------------------------------
        # TRAINING HISTORY
        # ----------------------------------------------------

        keras.callbacks.CSVLogger(

            str(
                output_dir
                / "history.csv"
            )
        ),
    ]


# ============================================================
# PREDICTION DECODER
# ============================================================

def decode_predictions(
    raw,
    ordinal_regression: bool,
):
    """
    Convert raw model output to predictions.
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
    Check whether model is collapsing to one class.
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
            "WARNING: Model predicts only one "
            "class in this batch."
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
    Evaluate model on untouched test dataset.
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

    # ========================================================
    # METRICS
    # ========================================================

    if output_dir is not None:

        evaluation_dir = (
            Path(output_dir)
            / "test_evaluation"
        )

        metrics = (
            save_evaluation_report(
                y_true,
                y_pred,
                y_prob,
                num_classes,
                str(evaluation_dir),
            )
        )

    else:

        from src.metrics import (
            compute_metrics,
        )

        metrics = compute_metrics(
            y_true,
            y_pred,
            num_classes,
        )

    # ========================================================
    # PRINT
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
# SAVE JSON
# ============================================================

def save_results(
    results: dict,
    output_dir: Path,
):
    """
    Save evaluation results to JSON.
    """

    path = (
        output_dir
        / "test_results.json"
    )

    serializable = {}

    for key, value in results.items():

        if key == "classification_report":
            continue

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
# BUILD DATASETS
# ============================================================

def create_datasets(
    dataset_root: Path,
    config: dict,
    num_classes: int,
):
    """
    Create train/validation/test datasets.
    """

    image_size = int(
        config.get(
            "image_size",
            224,
        )
    )

    batch_size = int(
        config.get(
            "batch_size",
            16,
        )
    )

    preprocess_cfg = {

        "use_roi": bool(
            config.get(
                "use_roi_detection",
                False,
            )
        ),

        "use_clahe": bool(
            config.get(
                "use_clahe",
                False,
            )
        ),

        "use_specular": bool(
            config.get(
                "use_specular_reduction",
                False,
            )
        ),

        "color_norm": bool(
            config.get(
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
    # TRAIN
    # ========================================================

    train_data = DentalCariesDataset(

        str(dataset_root),

        "train",

        image_size=image_size,

        batch_size=batch_size,

        augment=bool(
            config.get(
                "augmentation",
                True,
            )
        ),

        preprocess_cfg=preprocess_cfg,

        num_classes=num_classes,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_data = DentalCariesDataset(

        str(dataset_root),

        "val",

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

        "test",

        image_size=image_size,

        batch_size=batch_size,

        augment=False,

        preprocess_cfg=preprocess_cfg,

        num_classes=num_classes,
    )

    # ========================================================
    # VALIDATE CLASS DIRECTORIES
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
    # PRINT DISTRIBUTIONS
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

    train_ds = (
        train_data.as_tf_dataset(
            shuffle=True
        )
    )

    val_ds = (
        val_data.as_tf_dataset(
            shuffle=False
        )
    )

    test_ds = (
        test_data.as_tf_dataset(
            shuffle=False
        )
    )

    # ========================================================
    # IMPORTANT SCALE FIX
    # ========================================================

    logger.info(
        "Ensuring dataset images are "
        "float32 [0,255] for MobileNetV3 "
        "(no double scaling)."
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
# MAIN
# ============================================================

def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "Train ICDAS 0-4 classifier."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "ml/configs/default.yaml"
        ),
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

    num_classes = int(
        config.get(
            "num_classes",
            EXPECTED_NUM_CLASSES,
        )
    )

    ordinal_regression = bool(
        config.get(
            "ordinal_regression",
            False,
        )
    )

    # ========================================================
    # PROJECT PATHS
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    dataset_root = (
        project_root
        / config.get("dataset_root", "data/icdas")
    )

    models_root = (
        project_root
        / "models"
        / "icdas"
        / "current"
    )

    experiment_name = config.get(
        "experiment_name",
        "icdas_mobilenet_cbam_5class",
    )

    output_dir = (
        models_root
        / experiment_name
    )
    if config.get("output_dir"):
        output_dir = project_root / str(config["output_dir"])

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOG
    # ========================================================

    logger.info(
        "========================================"
    )

    logger.info(
        "ICDAS TRAINING"
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
        "Number of classes: %d",
        num_classes,
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        compute_class_weights_from_dataset(
            dataset_root,
            num_classes,
        )
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
    # STAGE 1
    # ========================================================

    logger.info(
        "========================================"
    )

    logger.info(
        "STAGE 1"
    )

    logger.info(
        "Training classifier head."
    )

    logger.info(
        "========================================"
    )

    model = build_model(

        num_classes=num_classes,

        image_size=int(
            config.get(
                "image_size",
                224,
            )
        ),

        attention_type=config.get(
            "use_attention",
            "cbam",
        ),

        dropout=float(
            config.get(
                "dropout",
                0.3,
            )
        ),

        pretrained=True,

        ordinal_regression=ordinal_regression,
    )

    model.summary()

    # ========================================================
    # COMPILE
    # ========================================================

    model = compile_model(

        model,

        config,

        learning_rate=float(
            config.get(
                "learning_rate",
                1e-4,
            )
        ),
    )

    # ========================================================
    # STAGE 1 TRAINING
    # ========================================================

    stage1_epochs = int(
        config.get(
            "head_epochs",
            15,
        )
    )

    logger.info(
        "Stage 1 epochs: %d",
        stage1_epochs,
    )

    model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=stage1_epochs,

        callbacks=get_callbacks(
            output_dir,
            config,
        ),

        class_weight=class_weights,

        verbose=1,
    )

    # ========================================================
    # BEST MODEL CHECK
    # ========================================================

    best_path = (
        output_dir
        / "best.keras"
    )

    if not best_path.exists():

        raise FileNotFoundError(
            f"Best model was not created: "
            f"{best_path}"
        )

    logger.info(
        "Best Stage 1 model: %s",
        best_path,
    )

    # ========================================================
    # LOAD BEST STAGE 1 MODEL
    # ========================================================

    model = keras.models.load_model(

        str(best_path),

        compile=False,

        custom_objects=(
            get_custom_objects()
        ),
    )

    # ========================================================
    # SANITY CHECK AFTER STAGE 1
    # ========================================================

    prediction_sanity_check(
        model,
        val_ds,
        ordinal_regression,
    )

    # ========================================================
    # STAGE 2
    # ========================================================

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

    unfreeze_top_layers(

        model,

        num_layers=int(
            config.get(
                "fine_tune_layers",
                20,
            )
        ),
    )

    model = compile_model(

        model,

        config,

        learning_rate=float(
            config.get(
                "fine_tune_learning_rate",
                1e-5,
            )
        ),
    )

    stage2_epochs = int(
        config.get(
            "fine_tune_epochs",
            25,
        )
    )

    logger.info(
        "Stage 2 epochs: %d",
        stage2_epochs,
    )

    model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=stage2_epochs,

        callbacks=get_callbacks(
            output_dir,
            config,
        ),

        class_weight=class_weights,

        verbose=1,
    )

    # ========================================================
    # LOAD FINAL BEST MODEL
    # ========================================================

    model = keras.models.load_model(

        str(best_path),

        compile=False,

        custom_objects=(
            get_custom_objects()
        ),
    )

    # ========================================================
    # FINAL SANITY CHECK
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

    overwrite_root = bool(
        config.get("overwrite_root_checkpoints", False)
    )
    deploy_path = (
        output_dir
        / "final.keras"
    )
    model.save(
        str(deploy_path)
    )
    root_best_path = None
    if overwrite_root:
        root_deploy = models_root / "deploy.keras"
        root_best_path = models_root / "best.keras"
        shutil.copy2(deploy_path, root_deploy)
        shutil.copy2(best_path, root_best_path)

    logger.info(
        "========================================"
    )

    logger.info(
        "MODEL SAVED"
    )

    logger.info(
        "Experiment best: %s",
        best_path,
    )

    logger.info(
        "Root best: %s",
        root_best_path,
    )

    logger.info(
        "Deployment: %s",
        deploy_path,
    )

    logger.info(
        "========================================"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

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
        "TRAINING COMPLETE"
    )

    logger.info(
        "========================================"


    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()