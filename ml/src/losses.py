"""
Loss functions and ordinal decode helpers for ICDAS classification.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .icdas import NUM_CLASSES


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        loss = y_true_one_hot * focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return loss_fn


def ordinal_loss(num_classes: int = NUM_CLASSES):
    """
    CORAL-style ordinal binary cross-entropy.

    For K classes there are K-1 thresholds. Target for threshold k is 1 if
    y_true > k, else 0.

    Example with ICDAS 0–4 (K=5, thresholds=4):
        ICDAS 0 -> [0, 0, 0, 0]
        ICDAS 1 -> [1, 0, 0, 0]
        ICDAS 2 -> [1, 1, 0, 0]
        ICDAS 3 -> [1, 1, 1, 0]
        ICDAS 4 -> [1, 1, 1, 1]
    """
    num_thresholds = num_classes - 1

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        losses = []
        for k in range(num_thresholds):
            target = tf.cast(y_true > k, tf.float32)
            prediction = y_pred[:, k]
            bce = tf.keras.backend.binary_crossentropy(target, prediction)
            losses.append(bce)
        losses = tf.stack(losses, axis=-1)
        return tf.reduce_mean(losses)

    return loss_fn


def ordinal_predict(ordinal_predictions):
    """
    Convert ordinal sigmoid outputs to class indices.

    Class = number of thresholds with P(y > k) >= 0.5.
    For 5 classes this uses 4 thresholds and yields grades 0–4.
    """
    predictions = tf.cast(ordinal_predictions >= 0.5, tf.int32)
    return tf.reduce_sum(predictions, axis=-1)


def ordinal_to_class_probabilities(ordinal_predictions) -> np.ndarray:
    """
    Convert P(y > k) outputs to a K-way probability vector.

        P(y = 0)     = 1 - P(y > 0)
        P(y = k)     = P(y > k-1) - P(y > k)
        P(y = K-1)   = P(y > K-2)

    Negative gaps from non-monotonic outputs are clipped, then renormalized.
    """
    p = np.asarray(ordinal_predictions, dtype=np.float32)
    if p.ndim == 1:
        p = p[np.newaxis, ...]
    p = np.clip(p, 0.0, 1.0)
    num_thresholds = p.shape[-1]
    num_classes = num_thresholds + 1
    probs = np.zeros((p.shape[0], num_classes), dtype=np.float32)
    probs[:, 0] = 1.0 - p[:, 0]
    for k in range(1, num_classes - 1):
        probs[:, k] = p[:, k - 1] - p[:, k]
    probs[:, -1] = p[:, -1]
    probs = np.maximum(probs, 0.0)
    totals = probs.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 0, 1.0, totals)
    probs = probs / totals
    return probs


def get_loss_function(loss_name: str, num_classes: int, class_weights=None):
    if loss_name == "ordinal":
        return ordinal_loss(num_classes)
    if loss_name == "focal":
        return focal_loss()
    return keras.losses.SparseCategoricalCrossentropy()
