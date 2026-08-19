"""
Loss functions for ICDAS classification.
"""

import tensorflow as tf
from tensorflow import keras


# ============================================================
# FOCAL LOSS
# ============================================================

def focal_loss(
    gamma: float = 2.0,
    alpha: float = 0.25,
):

    def loss_fn(
        y_true,
        y_pred,
    ):

        y_true = tf.cast(
            y_true,
            tf.int32,
        )

        y_true_one_hot = tf.one_hot(
            y_true,
            depth=tf.shape(y_pred)[-1],
        )

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7,
        )

        cross_entropy = (
            -y_true_one_hot
            * tf.math.log(y_pred)
        )

        focal_weight = tf.pow(
            1.0 - y_pred,
            gamma,
        )

        loss = (
            y_true_one_hot
            * focal_weight
            * cross_entropy
        )

        return tf.reduce_mean(
            tf.reduce_sum(
                loss,
                axis=-1,
            )
        )

    return loss_fn


# ============================================================
# ORDINAL REGRESSION LOSS
# ============================================================

def ordinal_loss(
    num_classes: int,
):

    num_thresholds = (
        num_classes - 1
    )

    def loss_fn(
        y_true,
        y_pred,
    ):

        y_true = tf.cast(
            y_true,
            tf.float32,
        )

        losses = []

        for k in range(
            num_thresholds
        ):

            # ------------------------------------------------
            # Target:
            #
            # ICDAS 0:
            # [0,0,0,0,0,0]
            #
            # ICDAS 1:
            # [1,0,0,0,0,0]
            #
            # ICDAS 2:
            # [1,1,0,0,0,0]
            #
            # ICDAS 6:
            # [1,1,1,1,1,1]
            # ------------------------------------------------

            target = tf.cast(
                y_true > k,
                tf.float32,
            )

            prediction = y_pred[:, k]

            bce = tf.keras.backend.binary_crossentropy(
                target,
                prediction,
            )

            losses.append(bce)

        losses = tf.stack(
            losses,
            axis=-1,
        )

        return tf.reduce_mean(
            losses
        )

    return loss_fn


# ============================================================
# ORDINAL CLASS DECODER
# ============================================================

def ordinal_predict(
    ordinal_predictions,
):
    """
    Convert 6 ordinal probabilities into
    ICDAS class 0-6.

    Example:

    [0.9, 0.8, 0.2, 0.1, 0.05, 0.01]

    -> 2

    because:

    P(y > 0) = 0.9
    P(y > 1) = 0.8
    P(y > 2) = 0.2
    ...
    """

    predictions = tf.cast(
        ordinal_predictions >= 0.5,
        tf.int32,
    )

    classes = tf.reduce_sum(
        predictions,
        axis=-1,
    )

    return classes


# ============================================================
# LOSS FACTORY
# ============================================================

def get_loss_function(
    loss_name: str,
    num_classes: int,
    class_weights=None,
):

    if loss_name == "ordinal":

        return ordinal_loss(
            num_classes
        )

    if loss_name == "focal":

        return focal_loss()

    if loss_name == "weighted_ce":

        return keras.losses.SparseCategoricalCrossentropy()

    return keras.losses.SparseCategoricalCrossentropy()