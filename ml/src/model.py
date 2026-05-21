"""
MobileNetV3-Small with CBAM attention and ordinal regression head.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers, applications

from .attention import add_attention


@keras.saving.register_keras_serializable(package="src.model")
class OrdinalRegressionHead(layers.Layer):
    """
    Ordinal regression via cumulative link model.
    Outputs K-1 threshold probabilities for K classes.
    """

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

    def build(self, input_shape):
        self.dense = layers.Dense(self.num_thresholds, activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)

    def predict_class(self, thresholds):
        """Convert threshold probabilities to class index."""
        # Class = number of thresholds exceeded
        return tf.reduce_sum(tf.cast(thresholds > 0.5, tf.int32), axis=-1)


def build_model(
    num_classes: int = 7,
    image_size: int = 224,
    attention_type: str = "cbam",
    ordinal: bool = True,
    dropout: float = 0.3,
    use_segmentation: bool = False,
) -> keras.Model:
    """
    Build ICDAS classification model.

    Args:
        num_classes: 7 for ICDAS 0-6, 5 for ICDAS 0-4
        attention_type: cbam | se | none
        ordinal: Use ordinal regression head
        use_segmentation: Add auxiliary segmentation decoder branch
    """
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")

    # MobileNetV3 Small backbone (ImageNet weights for transfer learning)
    backbone = applications.MobileNetV3Small(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    x = backbone.output

    if attention_type and attention_type != "none":
        x = add_attention(x, attention_type)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu", name="fc_hidden")(x)
    x = layers.Dropout(dropout * 0.5)(x)

    outputs = {}
    if ordinal and num_classes > 2:
        ordinal_out = OrdinalRegressionHead(num_classes, name="ordinal")(x)
        outputs["ordinal"] = ordinal_out
        # Also provide softmax for compatibility / ensemble
        class_out = layers.Dense(num_classes, activation="softmax", name="class")(x)
        outputs["class"] = class_out
    else:
        class_out = layers.Dense(num_classes, activation="softmax", name="class")(x)
        outputs["class"] = class_out

    if use_segmentation:
        # Lightweight decoder for lesion mask (multi-task)
        seg_features = backbone.get_layer(backbone.layers[-3].name).output
        seg = layers.UpSampling2D(4)(seg_features)
        seg = layers.Conv2D(64, 3, padding="same", activation="relu")(seg)
        seg = layers.UpSampling2D(4)(seg)
        seg = layers.Conv2D(1, 1, activation="sigmoid", name="segmentation")(seg)
        outputs["segmentation"] = seg

    model = keras.Model(inputs=inputs, outputs=outputs, name="icdas_mobilenet_cbam")
    return model


def ordinal_to_class(ordinal_probs: tf.Tensor) -> tf.Tensor:
    """Convert ordinal threshold probabilities to predicted class."""
    return tf.reduce_sum(tf.cast(ordinal_probs > 0.5, tf.int32), axis=-1)


def get_custom_objects() -> dict:
    """Custom layers required when loading saved Keras models."""
    from .attention import CBAM, ChannelAttention, SEBlock, SpatialAttention

    return {
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "SEBlock": SEBlock,
        "OrdinalRegressionHead": OrdinalRegressionHead,
    }


def get_last_conv_layer(model: keras.Model) -> layers.Layer:
    """Find last conv layer for Grad-CAM (MobileNetV3 block)."""
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, layers.Conv2D):
                    return sub
    # Fallback: use backbone output before GAP
    return model.get_layer("mobilenetv3small")
