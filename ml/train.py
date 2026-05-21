#!/usr/bin/env python3
"""
ICDAS model training script with stratified K-Fold, mixed precision, and early stopping.
Usage: python train.py --config configs/default.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

# Add ml directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import build_model, get_custom_objects, ordinal_to_class
from src.dataset import DentalCariesDataset
from src.losses import get_loss_function, ordinal_loss, focal_loss
from src.metrics import save_evaluation_report, compute_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_callbacks(output_dir: str, patience: int) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, "logs")),
        keras.callbacks.CSVLogger(os.path.join(output_dir, "history.csv")),
    ]


def get_lr_scheduler(config: dict, total_steps: int):
    """Cosine decay with optional warmup."""
    lr = config["learning_rate"]
    warmup = config.get("warmup_epochs", 5) * (total_steps // config["epochs"])

    def schedule(step):
        if step < warmup:
            return lr * (step / max(warmup, 1))
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return lr * 0.5 * (1 + np.cos(np.pi * progress))

    return keras.callbacks.LearningRateScheduler(
        lambda epoch, lr_val: float(schedule(epoch)), verbose=0
    )


def compile_model(model, config, class_weights=None):
    """Compile with appropriate losses for multi-output model."""
    num_classes = config["num_classes"]
    losses = {}
    loss_weights = {}
    metrics = {}

    if config.get("ordinal_regression") and "ordinal" in model.output_names:
        losses["ordinal"] = ordinal_loss(num_classes)
        loss_weights["ordinal"] = 1.0
        metrics["ordinal"] = "accuracy"
    if "class" in model.output_names:
        if config["loss"] == "focal":
            losses["class"] = focal_loss(
                config.get("focal_gamma", 2.0), config.get("focal_alpha", 0.25)
            )
        else:
            losses["class"] = keras.losses.SparseCategoricalCrossentropy()
        loss_weights["class"] = 0.5 if "ordinal" in losses else 1.0
        metrics["class"] = "accuracy"

    optimizer = keras.optimizers.AdamW(
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
    return model


def _multi_output_labels(ds, output_names):
    """Map scalar labels to a dict for multi-head models."""
    return ds.map(lambda x, y: (x, {name: y for name in output_names}))


def run_training_fold(model, train_ds, val_ds, config, output_dir, fold: int = 0):
    """Train a single cross-validation fold."""
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    callbacks = create_callbacks(fold_dir, config["early_stopping_patience"])

    if config.get("mixed_precision"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    output_names = list(model.output_names)
    if len(output_names) > 1:
        train_ds = _multi_output_labels(train_ds, output_names)
        val_ds = _multi_output_labels(val_ds, output_names)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=1,
    )
    return history, os.path.join(fold_dir, "best.keras")


def predict_classes(model, dataset, config) -> tuple:
    """Run inference and return y_true, y_pred, y_prob."""
    y_true, y_pred, y_prob = [], [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        if isinstance(preds, dict):
            if "ordinal" in preds:
                pred_cls = ordinal_to_class(preds["ordinal"]).numpy()
            else:
                pred_cls = np.argmax(preds["class"], axis=-1)
            prob = preds.get("class", preds["ordinal"])
        else:
            pred_cls = np.argmax(preds, axis=-1)
            prob = preds
        y_true.extend(labels.numpy())
        y_pred.extend(pred_cls)
        y_prob.extend(prob)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def main():
    parser = argparse.ArgumentParser(description="Train ICDAS caries detection model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=None, help="Single fold only")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = os.path.join(config["output_dir"], config["experiment_name"])
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    preprocess_cfg = {
        "use_roi": config.get("use_roi_detection", True),
        "use_clahe": config.get("use_clahe", True),
        "use_specular": config.get("use_specular_reduction", True),
        "color_norm": config.get("color_normalize", True),
    }

    # Load all training labels for stratified split
    train_data = DentalCariesDataset(
        config["dataset_root"],
        "train",
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        augment=False,
        preprocess_cfg=preprocess_cfg,
        annotations_file=config.get("annotations_file", "annotations.csv"),
    )
    labels = train_data.df["icdas_score"].values
    filenames = train_data.df["filename"].values

    k = config.get("k_folds", 5)
    min_class = int(train_data.df["icdas_score"].value_counts().min())
    if k > min_class:
        print(f"Warning: k_folds={k} exceeds smallest class ({min_class}); using k_folds={min_class}")
        k = max(min_class, 2)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    best_models = []

    splits = list(skf.split(filenames, labels))
    if args.fold is not None:
        splits = [splits[args.fold]]

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*50}\nFold {fold_idx + 1}/{k}\n{'='*50}")

        # Subset dataframes for this fold
        train_df = train_data.df.iloc[train_idx].copy()
        val_df = train_data.df.iloc[val_idx].copy()

        # Build datasets (simplified: reuse full loader with filtered df)
        train_split = DentalCariesDataset(
            config["dataset_root"], "train",
            image_size=config["image_size"], batch_size=config["batch_size"],
            augment=True, preprocess_cfg=preprocess_cfg,
        )
        train_split.df = train_df
        val_split = DentalCariesDataset(
            config["dataset_root"], "train",
            image_size=config["image_size"], batch_size=config["batch_size"],
            augment=False, preprocess_cfg=preprocess_cfg,
        )
        val_split.df = val_df

        model = build_model(
            num_classes=config["num_classes"],
            image_size=config["image_size"],
            attention_type=config.get("use_attention", "cbam"),
            ordinal=config.get("ordinal_regression", True),
            dropout=config.get("dropout", 0.3),
            use_segmentation=config.get("use_segmentation", False),
        )
        class_weights = None
        if config.get("class_weights"):
            class_weights = train_split.get_class_weights(config["num_classes"])

        model = compile_model(model, config, class_weights)
        print(model.summary())

        history, model_path = run_training_fold(
            model,
            train_split.as_tf_dataset(),
            val_split.as_tf_dataset(shuffle=False),
            config,
            output_dir,
            fold_idx,
        )
        best_models.append(model_path)

    # Final model: copy best fold to main output
    import shutil
    final_path = os.path.join(config["output_dir"], "best.keras")
    shutil.copy(best_models[-1], final_path)
    print(f"\nBest model saved to: {final_path}")

    # Evaluate on test set if available
    try:
        test_data = DentalCariesDataset(
            config["dataset_root"], "test",
            image_size=config["image_size"], batch_size=config["batch_size"],
            augment=False, preprocess_cfg=preprocess_cfg,
        )
        model = keras.models.load_model(
            final_path, compile=False, custom_objects=get_custom_objects()
        )
        y_true, y_pred, y_prob = predict_classes(model, test_data.as_tf_dataset(shuffle=False), config)
        metrics = save_evaluation_report(
            y_true, y_pred, y_prob, config["num_classes"],
            os.path.join(output_dir, "test_evaluation"),
        )
        print(f"\nTest metrics: {json.dumps({k: v for k, v in metrics.items() if k != 'confusion_matrix'}, indent=2)}")
    except FileNotFoundError:
        print("No test split found — skipping test evaluation.")


if __name__ == "__main__":
    main()
