"""
Loss functions: focal loss, weighted CE, ordinal regression loss.
"""

import tensorflow as tf
from tensorflow import keras


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Multi-class focal loss for class imbalance."""

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot + (1 - alpha) * (1 - y_true_one_hot)
        focal_weight = weight * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * cross_entropy, axis=-1))

    return loss_fn


def ordinal_loss(num_classes: int):
    """
    Ordinal regression loss using cumulative link model.
    y_true: integer class 0..K-1
    y_pred: K-1 sigmoid thresholds
    """

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        num_thresholds = num_classes - 1
        losses = []
        for k in range(num_thresholds):
            # Target: 1 if true class > k, else 0
            target_k = tf.cast(y_true > k, tf.float32)
            pred_k = y_pred[:, k]
            pred_k = tf.clip_by_value(pred_k, 1e-7, 1.0 - 1e-7)
            bce = -(target_k * tf.math.log(pred_k) + (1 - target_k) * tf.math.log(1 - pred_k))
            losses.append(bce)
        return tf.reduce_mean(tf.add_n(losses))

    return loss_fn


def get_loss_function(loss_name: str, num_classes: int, class_weights=None):
    """Factory for loss functions."""
    if loss_name == "focal":
        return focal_loss()
    if loss_name == "ordinal":
        return ordinal_loss(num_classes)
    if loss_name == "weighted_ce" and class_weights is not None:
        return keras.losses.SparseCategoricalCrossentropy()
    return keras.losses.SparseCategoricalCrossentropy()
