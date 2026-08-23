#!/usr/bin/env python3
"""
ICDAS training pipeline.

Supports:
    - ICDAS 0–4 (5 classes)
    - Standard 5-class classification
    - Optional ordinal regression
    - MobileNetV3-Small
    - CBAM attention
    - Training-only augmentation
    - Class-weighted training
    - Early stopping
    - Learning-rate reduction
    - Full classification metrics
    - Confusion matrix

IMPORTANT:
    Class weights are calculated from dataset/train/0..4
    to reduce bias toward the majority ICDAS classes.
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

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

from src.dataset import DentalCariesDataset
from src.icdas import NUM_CLASSES
from src.losses import (
    ordinal_loss,
    ordinal_predict,
    ordinal_to_class_probabilities,
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
# CONFIG
# ============================================================

def load_config(path: str):
    """Load YAML configuration file."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights_from_dataset(
    dataset_root: Path,
    num_classes: int,
):
    """
    Calculate balanced class weights from:

        dataset/train/0
        dataset/train/1
        dataset/train/2
        dataset/train/3
        dataset/train/4

    Formula:

        weight = total_samples / (num_classes * class_count)

    This gives lower weight to majority classes and higher weight
    to minority classes.
    """

    train_root = dataset_root / "train"

    counts = {}

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    logger.info("Calculating class weights from: %s", train_root)

    for class_id in range(num_classes):

        class_dir = train_root / str(class_id)

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Training class directory does not exist: {class_dir}"
            )

        count = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in valid_extensions
        )

        if count == 0:
            raise ValueError(
                f"Class {class_id} has zero training images."
            )

        counts[class_id] = count

    total_samples = sum(counts.values())

    class_weights = {}

    for class_id in range(num_classes):

        weight = total_samples / (
            num_classes * counts[class_id]
        )

        class_weights[class_id] = float(weight)

    logger.info("Training distribution:")

    for class_id in range(num_classes):
        logger.info(
            "ICDAS %d: %d images -> weight %.4f",
            class_id,
            counts[class_id],
            class_weights[class_id],
        )

    return class_weights


# ============================================================
# MODEL COMPILATION
# ============================================================

def compile_model(
    model,
    config,
    learning_rate,
):
    """
    Compile the model according to YAML configuration.
    """

    ordinal_regression = config.get(
        "ordinal_regression",
        False,
    )

    loss_name = config.get(
        "loss",
        "sparse_categorical_crossentropy",
    )

    num_classes = int(
        config.get(
            "num_classes",
            NUM_CLASSES,
        )
    )

    # --------------------------------------------------------
    # ORDINAL REGRESSION
    # --------------------------------------------------------

    if ordinal_regression or loss_name == "ordinal":

        loss = ordinal_loss(num_classes)

        logger.info(
            "Loss: ordinal regression "
            "(%s classes, %s thresholds)",
            num_classes,
            num_classes - 1,
        )

    # --------------------------------------------------------
    # FOCAL LOSS
    # --------------------------------------------------------

    elif loss_name == "focal":

        from src.losses import focal_loss

        loss = focal_loss(
            gamma=config.get(
                "focal_gamma",
                2.0,
            ),
            alpha=config.get(
                "focal_alpha",
                0.25,
            ),
        )

        logger.info("Loss: focal")

    # --------------------------------------------------------
    # STANDARD CLASSIFICATION
    # --------------------------------------------------------

    else:

        loss = keras.losses.SparseCategoricalCrossentropy()

        logger.info(
            "Loss: sparse categorical cross entropy"
        )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=config.get(
            "weight_decay",
            1e-4,
        ),
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    return model


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(output_dir):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    return [

        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                output_dir,
                "best.keras",
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),

        keras.callbacks.CSVLogger(
            os.path.join(
                output_dir,
                "history.csv",
            )
        ),
    ]


# ============================================================
# PREDICTION DECODER
# ============================================================

def _decode_predictions(
    raw,
    ordinal_regression: bool,
):
    """
    Convert model output into predicted class and probabilities.
    """

    raw = np.asarray(raw)

    if ordinal_regression:

        preds = ordinal_predict(
            tf.convert_to_tensor(raw)
        ).numpy()

        probs = ordinal_to_class_probabilities(
            raw
        )

        return preds, probs

    preds = np.argmax(
        raw,
        axis=-1,
    )

    return preds, raw


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataset,
    num_classes,
    ordinal_regression=False,
    output_dir: Path | None = None,
):
    """
    Evaluate the model on the untouched test dataset.
    """

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in dataset:

        probabilities = model.predict(
            images,
            verbose=0,
        )

        preds, probs = _decode_predictions(
            probabilities,
            ordinal_regression,
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            preds
        )

        y_prob.extend(
            probs
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

    # --------------------------------------------------------
    # SAVE EVALUATION REPORT
    # --------------------------------------------------------

    eval_dir = None

    if output_dir is not None:

        eval_dir = str(
            Path(output_dir)
            / "test_evaluation"
        )

    if eval_dir:

        metrics = save_evaluation_report(
            y_true,
            y_pred,
            y_prob,
            num_classes,
            eval_dir,
        )

    else:

        from src.metrics import compute_metrics

        metrics = compute_metrics(
            y_true,
            y_pred,
            num_classes,
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        f"TEST ACCURACY: "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"MACRO F1:      "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"WEIGHTED F1:   "
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )

    print(
        "==============================\n"
    )

    print(
        "PER-CLASS RESULTS:"
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
            metrics["confusion_matrix"]
        )
    )

    print(
        "\n"
        + metrics[
            "classification_report"
        ]
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENTS
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="ml/configs/default.yaml",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    num_classes = int(
        config.get(
            "num_classes",
            NUM_CLASSES,
        )
    )

    if num_classes != NUM_CLASSES:

        logger.warning(
            "Config num_classes=%s differs "
            "from project default %s",
            num_classes,
            NUM_CLASSES,
        )

    ordinal_regression = bool(
        config.get(
            "ordinal_regression",
            False,
        )
    )

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    dataset_root = (
        project_root
        / "dataset"
    )

    models_root = (
        project_root
        / "models"
    )

    output_dir = (
        models_root
        / config.get(
            "experiment_name",
            "icdas_model",
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOG CONFIG
    # --------------------------------------------------------

    logger.info(
        "Dataset: %s",
        dataset_root,
    )

    logger.info(
        "Output: %s",
        output_dir,
    )

    logger.info(
        "Classes: %s (ICDAS 0–%s)",
        num_classes,
        num_classes - 1,
    )

    logger.info(
        "Ordinal regression: %s",
        ordinal_regression,
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = (
        compute_class_weights_from_dataset(
            dataset_root,
            num_classes,
        )
    )

    logger.info(
        "Class weighting ENABLED."
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    preprocess_cfg = {

        "use_roi": config.get(
            "use_roi_detection",
            True,
        ),

        "use_clahe": config.get(
            "use_clahe",
            True,
        ),

        "use_specular": config.get(
            "use_specular_reduction",
            True,
        ),

        "color_norm": config.get(
            "color_normalize",
            True,
        ),
    }

    # --------------------------------------------------------
    # TRAIN DATASET
    # --------------------------------------------------------

    train_data = DentalCariesDataset(

        str(dataset_root),

        "train",

        image_size=config[
            "image_size"
        ],

        batch_size=config[
            "batch_size"
        ],

        augment=bool(
            config.get(
                "augmentation",
                True,
            )
        ),

        preprocess_cfg=preprocess_cfg,

        num_classes=num_classes,
    )

    # --------------------------------------------------------
    # VALIDATION DATASET
    # --------------------------------------------------------

    val_data = DentalCariesDataset(

        str(dataset_root),

        "val",

        image_size=config[
            "image_size"
        ],

        batch_size=config[
            "batch_size"
        ],

        augment=False,

        preprocess_cfg=preprocess_cfg,

        num_classes=num_classes,
    )

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    test_data = DentalCariesDataset(

        str(dataset_root),

        "test",

        image_size=config[
            "image_size"
        ],

        batch_size=config[
            "batch_size"
        ],

        augment=False,

        preprocess_cfg=preprocess_cfg,

        num_classes=num_classes,
    )

    # --------------------------------------------------------
    # VALIDATE DATASETS
    # --------------------------------------------------------

    train_data.validate_classes(
        num_classes
    )

    val_data.validate_classes(
        num_classes
    )

    test_data.validate_classes(
        num_classes
    )

    # --------------------------------------------------------
    # PRINT DISTRIBUTIONS
    # --------------------------------------------------------

    train_data.print_distribution(
        num_classes
    )

    val_data.print_distribution(
        num_classes
    )

    test_data.print_distribution(
        num_classes
    )

    # --------------------------------------------------------
    # TF DATASETS
    # --------------------------------------------------------

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
    # STAGE 1
    # ========================================================

    logger.info(
        "STAGE 1: train classifier head"
    )

    model = build_model(

        num_classes=num_classes,

        image_size=config[
            "image_size"
        ],

        attention_type=config.get(
            "use_attention",
            "cbam",
        ),

        dropout=config.get(
            "dropout",
            0.3,
        ),

        pretrained=True,

        ordinal_regression=ordinal_regression,
    )

    model.summary()

    model = compile_model(

        model,

        config,

        learning_rate=config.get(
            "learning_rate",
            1e-4,
        ),
    )

    logger.info(
        "Starting Stage 1 with class weights: %s",
        class_weights,
    )

    model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=config.get(
            "head_epochs",
            30,
        ),

        callbacks=get_callbacks(
            output_dir
        ),

        class_weight=class_weights,

        verbose=1,
    )

    # ========================================================
    # LOAD BEST STAGE 1 MODEL
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

    model = keras.models.load_model(

        best_path,

        compile=False,

        custom_objects=get_custom_objects(),
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    logger.info(
        "STAGE 2: fine-tune MobileNet top layers"
    )

    unfreeze_top_layers(

        model,

        num_layers=config.get(
            "fine_tune_layers",
            30,
        ),
    )

    model = compile_model(

        model,

        config,

        learning_rate=config.get(
            "fine_tune_learning_rate",
            1e-5,
        ),
    )

    logger.info(
        "Starting Stage 2 with class weights: %s",
        class_weights,
    )

    model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=config.get(
            "fine_tune_epochs",
            40,
        ),

        callbacks=get_callbacks(
            output_dir
        ),

        class_weight=class_weights,

        verbose=1,
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    model = keras.models.load_model(

        best_path,

        compile=False,

        custom_objects=get_custom_objects(),
    )

    # ========================================================
    # SAVE DEPLOYMENT MODEL
    # ========================================================

    deploy_path = (
        models_root
        / "deploy.keras"
    )

    root_best = (
        models_root
        / "best.keras"
    )

    model.save(
        deploy_path
    )

    shutil.copy2(
        best_path,
        root_best,
    )

    logger.info(
        "Saved 5-class checkpoints: "
        "%s and %s",
        root_best,
        deploy_path,
    )

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    results = evaluate_model(

        model,

        test_ds,

        num_classes,

        ordinal_regression=ordinal_regression,

        output_dir=output_dir,
    )

    # ========================================================
    # SAVE RESULTS JSON
    # ========================================================

    with open(
        output_dir
        / "test_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(

            {
                k: v
                for k, v in results.items()
                if k != "classification_report"
            },

            f,

            indent=2,
        )

        f.write("\n")

    logger.info(
        "Training complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()