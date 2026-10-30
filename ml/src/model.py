"""
MobileNetV3Small + CBAM model for ICDAS 0-4 classification.

V6:
- 5 classes
- MobileNetV3Small
- ImageNet pretrained backbone
- CBAM attention
- Softmax classification
- Optional ordinal regression retained for future experiments
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small


logger = logging.getLogger("icdas.model")


# ============================================================
# CBAM CHANNEL ATTENTION
# ============================================================

class ChannelAttention(layers.Layer):
    """
    Channel Attention Module from CBAM.
    """

    def __init__(
        self,
        ratio: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ratio = ratio

        self.avg_pool = layers.GlobalAveragePooling2D(
            keepdims=True
        )

        self.max_pool = layers.GlobalMaxPooling2D(
            keepdims=True
        )

        self.shared_dense_1 = None
        self.shared_dense_2 = None

    def build(self, input_shape):
        channels = int(input_shape[-1])

        hidden_channels = max(
            channels // self.ratio,
            1,
        )

        self.shared_dense_1 = layers.Conv2D(
            hidden_channels,
            kernel_size=1,
            activation="relu",
            padding="same",
        )

        self.shared_dense_2 = layers.Conv2D(
            channels,
            kernel_size=1,
            padding="same",
        )

        super().build(input_shape)

    def call(self, inputs):
        avg = self.avg_pool(inputs)
        max_value = self.max_pool(inputs)

        avg = self.shared_dense_2(
            self.shared_dense_1(avg)
        )

        max_value = self.shared_dense_2(
            self.shared_dense_1(max_value)
        )

        attention = tf.nn.sigmoid(
            avg + max_value
        )

        return inputs * attention


# ============================================================
# CBAM SPATIAL ATTENTION
# ============================================================

class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module from CBAM.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size

        self.conv = layers.Conv2D(
            1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
        )

    def call(self, inputs):
        avg = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True,
        )

        max_value = tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True,
        )

        combined = tf.concat(
            [
                avg,
                max_value,
            ],
            axis=-1,
        )

        attention = self.conv(
            combined
        )

        return inputs * attention


# ============================================================
# CBAM
# ============================================================

class CBAM(layers.Layer):
    """
    Convolutional Block Attention Module.
    """

    def __init__(
        self,
        ratio: int = 8,
        spatial_kernel: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.channel_attention = ChannelAttention(
            ratio=ratio,
            name="channel_attention",
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel,
            name="spatial_attention",
        )

    def call(self, inputs):
        x = self.channel_attention(
            inputs
        )

        x = self.spatial_attention(
            x
        )

        return x


# ============================================================
# MODEL BUILD
# ============================================================

def build_model(
    num_classes: int = 5,
    image_size: int = 224,
    attention_type: str = "cbam",
    dropout: float = 0.30,
    pretrained: bool = True,
    ordinal_regression: bool = False,
):
    """
    Build MobileNetV3Small + CBAM.

    Input:
        RGB float32 [0,255]

    MobileNetV3Small:
        include_preprocessing=True

    Therefore the model itself performs the expected
    MobileNetV3 preprocessing.
    """

    if num_classes != 5:
        raise ValueError(
            "This project supports ICDAS 0-4 only. "
            "num_classes must be 5."
        )

    input_shape = (
        image_size,
        image_size,
        3,
    )

    # ========================================================
    # INPUT
    # ========================================================

    inputs = keras.Input(
        shape=input_shape,
        name="image",
    )

    # ========================================================
    # BACKBONE
    # ========================================================

    weights = (
        "imagenet"
        if pretrained
        else None
    )

    backbone = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling=None,
        include_preprocessing=True,
        name="mobilenetv3small",
    )

    # Stage 1:
    # freeze entire backbone

    backbone.trainable = False

    x = backbone(
        inputs,
        training=False,
    )

    # ========================================================
    # ATTENTION
    # ========================================================

    if (
        attention_type
        and attention_type.lower()
        == "cbam"
    ):

        x = CBAM(
            name="cbam",
        )(x)

    # ========================================================
    # GLOBAL POOLING
    # ========================================================

    x = layers.GlobalAveragePooling2D(
        name="gap",
    )(x)

    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.Dropout(
        dropout,
        name="dropout_1",
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(
            1e-4
        ),
        name="fc_hidden",
    )(x)

    x = layers.Dropout(
        dropout * 0.5,
        name="dropout_2",
    )(x)

    # ========================================================
    # ORDINAL HEAD
    # ========================================================

    if ordinal_regression:

        outputs = layers.Dense(
            num_classes - 1,
            activation="sigmoid",
            kernel_regularizer=keras.regularizers.l2(
                1e-4
            ),
            name="ordinal",
        )(x)

    # ========================================================
    # SOFTMAX HEAD
    # ========================================================

    else:

        outputs = layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=keras.regularizers.l2(
                1e-4
            ),
            name="class",
        )(x)

    # ========================================================
    # MODEL
    # ========================================================

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="icdas_mobilenet_cbam",
    )

    return model


# ============================================================
# UNFREEZE TOP BACKBONE LAYERS
# ============================================================

def unfreeze_top_layers(
    model,
    num_layers: int = 20,
):
    """
    Unfreeze the top N backbone layers.

    BatchNormalization layers remain frozen because
    small dental datasets can make BatchNorm statistics
    unstable during fine-tuning.
    """

    backbone = model.get_layer(
        "mobilenetv3small"
    )

    # Freeze everything first.

    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    # Select last N layers.

    selected_layers = (
        backbone.layers[-num_layers:]
    )

    trainable_count = 0

    for layer in selected_layers:

        if isinstance(
            layer,
            layers.BatchNormalization,
        ):
            layer.trainable = False

        else:
            layer.trainable = True
            trainable_count += 1

    logger.info(
        "Unfroze top %d MobileNetV3 layers "
        "(%d trainable layers).",
        num_layers,
        trainable_count,
    )


# ============================================================
# CUSTOM OBJECTS
# ============================================================

def get_custom_objects():
    """
    Custom Keras objects required when loading the model.
    """

    return {
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
    }


# ============================================================
# DEBUG
# ============================================================

def debug_backbone_layers(
    model,
):
    """
    Print MobileNetV3 layer information.
    """

    backbone = model.get_layer(
        "mobilenetv3small"
    )

    logger.info(
        "MobileNetV3 layer count: %d",
        len(backbone.layers),
    )

    for index, layer in enumerate(
        backbone.layers
    ):

        logger.info(
            "%d: %-40s trainable=%s",
            index,
            layer.name,
            layer.trainable,
        )


# ============================================================
# LAST CONVOLUTIONAL LAYER
# ============================================================

def get_last_conv_layer(
    model,
):
    """
    Find the last Conv2D layer.

    Used later by Grad-CAM.
    """

    for layer in reversed(
        model.layers
    ):

        if isinstance(
            layer,
            layers.Conv2D,
        ):
            return layer.name

    backbone = model.get_layer(
        "mobilenetv3small"
    )

    for layer in reversed(
        backbone.layers
    ):

        if isinstance(
            layer,
            layers.Conv2D,
        ):
            return layer.name

    raise ValueError(
        "No Conv2D layer found for Grad-CAM."
    )