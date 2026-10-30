"""
Loss functions and ordinal decode helpers for ICDAS classification.

CHANGES vs your original:
1. focal_loss() now actually uses `alpha` (it was accepted but silently
   ignored before - class-balance weighting was doing nothing).
2. focal_loss() and ordinal_loss() now accept an optional
   `class_weights` array so they can carry per-class weighting
   internally too (belt-and-suspenders with Keras' class_weight=
   in model.fit(), which only reliably applies to the loss for
   standard losses - custom losses can silently ignore it depending
   on Keras version, so it's safer to also weight inside the loss).
3. Added QuadraticWeightedKappa metric class - see train.py patch
   notes for how to wire it in. This is what will actually let you
   see whether Stage 2 fine-tuning is doing anything, since val_loss
   alone can look flat while the model is just getting more confident
   about predicting class 0.

Function names and signatures are unchanged from your original file,
so this is a drop-in replacement - no import changes needed in train.py.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .icdas import NUM_CLASSES


def focal_loss(gamma: float = 2.0, alpha: float = 0.25, class_weights=None):
    """
    Standard multi-class focal loss:
        L = -alpha * (1 - pt)^gamma * log(pt)

    alpha: global weight applied to every class's loss term. Keep the
        default 0.25 unless you have a specific reason to change it -
        it's a mild dampener on the overall loss magnitude, separate
        from per-class balancing.
    class_weights: optional array of length num_classes, e.g. the same
        weights your compute_class_weights_from_dataset() already
        prints in your logs: [0.7902, 1.0367, 1.0762, 1.1078, 1.0660].
        Passing them here (in addition to model.fit(class_weight=...))
        guarantees the weighting is applied regardless of Keras version
        quirks with custom loss functions.
    """
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true_idx = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true_idx, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        loss = alpha * y_true_one_hot * focal_weight * cross_entropy
        per_sample = tf.reduce_sum(loss, axis=-1)

        if class_weights_tensor is not None:
            sample_weights = tf.gather(class_weights_tensor, y_true_idx)
            per_sample = per_sample * sample_weights

        return tf.reduce_mean(per_sample)

    return loss_fn


def ordinal_loss(num_classes: int = NUM_CLASSES, class_weights=None):
    """
    CORAL-style ordinal binary cross-entropy.

    For K classes there are K-1 thresholds. Target for threshold k is 1 if
    y_true > k, else 0.

    Example with ICDAS 0-4 (K=5, thresholds=4):
        ICDAS 0 -> [0, 0, 0, 0]
        ICDAS 1 -> [1, 0, 0, 0]
        ICDAS 2 -> [1, 1, 0, 0]
        ICDAS 3 -> [1, 1, 1, 0]
        ICDAS 4 -> [1, 1, 1, 1]

    class_weights: optional, same as focal_loss above - applied per
    sample based on the sample's true integer class.
    """
    num_thresholds = num_classes - 1
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true_idx = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_f = tf.cast(y_true_idx, tf.float32)

        losses = []
        for k in range(num_thresholds):
            target = tf.cast(y_true_f > k, tf.float32)
            prediction = y_pred[:, k]
            bce = tf.keras.backend.binary_crossentropy(target, prediction)
            losses.append(bce)
        losses = tf.stack(losses, axis=-1)
        per_sample = tf.reduce_mean(losses, axis=-1)

        if class_weights_tensor is not None:
            sample_weights = tf.gather(class_weights_tensor, y_true_idx)
            per_sample = per_sample * sample_weights

        return tf.reduce_mean(per_sample)

    return loss_fn


def ordinal_predict(ordinal_predictions):
    """
    Convert ordinal sigmoid outputs to class indices.

    Class = number of thresholds with P(y > k) >= 0.5.
    For 5 classes this uses 4 thresholds and yields grades 0-4.
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
        return ordinal_loss(num_classes, class_weights=class_weights)
    if loss_name == "focal":
        return focal_loss(class_weights=class_weights)
    return keras.losses.SparseCategoricalCrossentropy()


# ============================================================
# QUADRATIC WEIGHTED KAPPA - wire this into compile_model() metrics
# ============================================================

class QuadraticWeightedKappa(keras.metrics.Metric):
    """
    QWK for ordinal classification. Works with either softmax
    (num_classes outputs) or ordinal (num_classes-1 sigmoid outputs) -
    auto-detects based on output width.

    Accuracy and val_loss both currently treat "predicted 0, true 4" the
    same severity as "predicted 3, true 4" (both just "wrong"). QWK
    penalizes proportional to how far off the prediction is, which is
    how ICDAS grading is actually judged, and can't be gamed by a model
    that just predicts the majority class more confidently.

    Use this as your ModelCheckpoint/EarlyStopping monitor instead of
    val_loss - see train.py patch notes.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, name: str = "qwk", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_mat = self.add_weight(
            name="conf_mat",
            shape=(num_classes, num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        if y_pred.shape[-1] == self.num_classes:
            y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        else:
            y_pred_labels = tf.cast(
                tf.reduce_sum(tf.cast(y_pred >= 0.5, tf.int32), axis=-1),
                tf.int32,
            )

        batch_conf = tf.math.confusion_matrix(
            y_true, y_pred_labels,
            num_classes=self.num_classes,
            dtype=tf.float32,
        )
        self.conf_mat.assign_add(batch_conf)

    def result(self):
        cm = self.conf_mat
        n = self.num_classes

        weights = tf.constant(
            [[(i - j) ** 2 for j in range(n)] for i in range(n)],
            dtype=tf.float32,
        )

        row_sums = tf.reduce_sum(cm, axis=1)
        col_sums = tf.reduce_sum(cm, axis=0)
        total = tf.reduce_sum(cm)

        expected = tf.tensordot(row_sums, col_sums, axes=0) / tf.maximum(total, 1.0)

        observed_weighted = tf.reduce_sum(weights * cm)
        expected_weighted = tf.reduce_sum(weights * expected)

        kappa = 1.0 - observed_weighted / tf.maximum(expected_weighted, 1e-7)
        return kappa

    def reset_state(self):
        self.conf_mat.assign(tf.zeros((self.num_classes, self.num_classes)))