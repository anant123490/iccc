#!/usr/bin/env python3

"""
ICDAS training pipeline.

Stage 1:
    Train classifier head with frozen MobileNetV3.

Stage 2:
    Fine-tune top MobileNet layers.

Stage 3:
    Evaluate on untouched test set.

Supports:
    - ICDAS 0-6
    - Ordinal regression
    - MobileNetV3-Small
    - CBAM
    - Data augmentation
    - Early stopping
    - Learning-rate reduction
    - Confusion matrix
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from tensorflow import keras


# ============================================================
# PATH
# ============================================================

ML_DIR = Path(
    __file__
).resolve().parent

sys.path.insert(
    0,
    str(ML_DIR),
)


# ============================================================
# IMPORTS
# ============================================================

from src.model import (
    build_model,
    unfreeze_top_layers,
    get_custom_objects,
)

from src.dataset import (
    DentalCariesDataset,
)

from src.losses import (
    focal_loss,
    ordinal_loss,
    ordinal_predict,
)


# ============================================================
# CONFIG
# ============================================================

def load_config(
    path: str,
):

    with open(
        path,
        "r",
    ) as f:

        return yaml.safe_load(f)


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_model(
    model,
    config,
    learning_rate,
):

    ordinal_regression = config.get(
        "ordinal_regression",
        True,
    )

    loss_name = config.get(
        "loss",
        "ordinal",
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    if ordinal_regression:

        loss = ordinal_loss(
            config["num_classes"]
        )

        print(
            "\nLOSS: ORDINAL REGRESSION"
        )

    elif loss_name == "focal":

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

        print(
            "\nLOSS: FOCAL"
        )

    else:

        loss = (
            keras.losses
            .SparseCategoricalCrossentropy()
        )

        print(
            "\nLOSS: "
            "SPARSE CATEGORICAL "
            "CROSS ENTROPY"
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

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    return model


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(
    output_dir,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    return [

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),

        # ----------------------------------------------------
        # Reduce learning rate
        # ----------------------------------------------------

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),

        # ----------------------------------------------------
        # Save training history
        # ----------------------------------------------------

        keras.callbacks.CSVLogger(
            os.path.join(
                output_dir,
                "history.csv",
            )
        ),
    ]


# ============================================================
# ORDINAL EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataset,
    num_classes,
):

    y_true = []

    y_pred = []

    y_prob = []

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    for images, labels in dataset:

        probabilities = model.predict(
            images,
            verbose=0,
        )

        predictions = (
            ordinal_predict(
                tf.convert_to_tensor(
                    probabilities
                )
            )
            .numpy()
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predictions
        )

        y_prob.extend(
            probabilities
        )

    # --------------------------------------------------------
    # NUMPY
    # --------------------------------------------------------

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
    # ACCURACY
    # --------------------------------------------------------

    accuracy = np.mean(
        y_true == y_pred
    )

    print(
        "\n=============================="
    )

    print(
        "TEST ACCURACY:",
        f"{accuracy * 100:.2f}%"
    )

    print(
        "==============================\n"
    )

    # --------------------------------------------------------
    # PER CLASS
    # --------------------------------------------------------

    print(
        "PER CLASS RESULTS:"
    )

    for c in range(
        num_classes
    ):

        mask = (
            y_true == c
        )

        count = np.sum(
            mask
        )

        if count == 0:

            continue

        class_acc = np.mean(
            y_pred[mask] == c
        )

        print(
            f"Grade {c}: "
            f"{class_acc * 100:.2f}% "
            f"({count} samples)"
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print(
        "\nCONFUSION MATRIX:"
    )

    cm = (
        tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=num_classes,
        )
        .numpy()
    )

    print(cm)

    return {

        "accuracy": float(
            accuracy
        ),

        "confusion_matrix":
            cm.tolist(),
    }


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
        default=(
            "ml/configs/default.yaml"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    config = load_config(
        args.config
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

    output_dir = (
        project_root
        / "models"
        / config.get(
            "experiment_name",
            "icdas_model",
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nDataset:",
        dataset_root,
    )

    print(
        "Output:",
        output_dir,
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

        "use_specular":
            config.get(
                "use_specular_reduction",
                True,
            ),

        "color_norm":
            config.get(
                "color_normalize",
                True,
            ),
    }

    # --------------------------------------------------------
    # DATASETS
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
        augment=True,
        preprocess_cfg=preprocess_cfg,
    )

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
    )

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
    )

    # --------------------------------------------------------
    # VALIDATE CLASSES
    # --------------------------------------------------------

    num_classes = config[
        "num_classes"
    ]

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
    # PRINT DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nTRAIN DISTRIBUTION:"
    )

    train_data.print_distribution(
        num_classes
    )

    print(
        "\nVALIDATION DISTRIBUTION:"
    )

    val_data.print_distribution(
        num_classes
    )

    print(
        "\nTEST DISTRIBUTION:"
    )

    test_data.print_distribution(
        num_classes
    )

    # --------------------------------------------------------
    # TF DATASETS
    # --------------------------------------------------------

    train_ds = (
        train_data
        .as_tf_dataset(
            shuffle=True
        )
    )

    val_ds = (
        val_data
        .as_tf_dataset(
            shuffle=False
        )
    )

    test_ds = (
        test_data
        .as_tf_dataset(
            shuffle=False
        )
    )

    # ========================================================
    # STAGE 1
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "STAGE 1: TRAIN CLASSIFIER"
    )

    print(
        "================================\n"
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
        ordinal_regression=config.get(
            "ordinal_regression",
            True,
        ),
    )

    # --------------------------------------------------------
    # SHOW MODEL
    # --------------------------------------------------------

    model.summary()

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model = compile_model(
        model,
        config,
        learning_rate=config.get(
            "learning_rate",
            1e-4,
        ),
    )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    callbacks = get_callbacks(
        output_dir
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.get(
            "head_epochs",
            20,
        ),
        callbacks=callbacks,
        verbose=1,
    )

    # ========================================================
    # LOAD BEST STAGE 1
    # ========================================================

    best_path = (
        output_dir
        / "best.keras"
    )

    model = keras.models.load_model(
        best_path,
        compile=False,
        custom_objects=
            get_custom_objects(),
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "STAGE 2: FINE TUNING"
    )

    print(
        "================================\n"
    )

    unfreeze_top_layers(
        model,
        num_layers=config.get(
            "fine_tune_layers",
            30,
        ),
    )

    # --------------------------------------------------------
    # RECOMPILE
    # --------------------------------------------------------

    model = compile_model(
        model,
        config,
        learning_rate=config.get(
            "fine_tune_learning_rate",
            1e-5,
        ),
    )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    callbacks = get_callbacks(
        output_dir
    )

    # --------------------------------------------------------
    # FINE-TUNE
    # --------------------------------------------------------

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.get(
            "fine_tune_epochs",
            30,
        ),
        callbacks=callbacks,
        verbose=1,
    )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    model = keras.models.load_model(
        best_path,
        compile=False,
        custom_objects=
            get_custom_objects(),
    )

    final_path = (
        project_root
        / "models"
        / "deploy.keras"
    )

    model.save(
        final_path
    )

    print(
        "\nFINAL MODEL:"
    )

    print(
        final_path
    )

    # ========================================================
    # TEST
    # ========================================================

    results = evaluate_model(
        model,
        test_ds,
        num_classes,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        output_dir
        / "test_results.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\nTraining complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()