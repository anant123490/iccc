"""
Attention mechanisms: CBAM (Convolutional Block Attention Module) and SE (Squeeze-Excitation).
"""

import keras
import tensorflow as tf
from keras import layers


@keras.saving.register_keras_serializable(package="src")
class ChannelAttention(layers.Layer):
    """Channel attention sub-module for CBAM."""

    def __init__(self, ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense_one = layers.Dense(
            channels // self.ratio, activation="relu", use_bias=False
        )
        self.shared_dense_two = layers.Dense(channels, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))
        attention = tf.nn.sigmoid(avg_out + max_out)
        return inputs * attention


@keras.saving.register_keras_serializable(package="src")
class SpatialAttention(layers.Layer):
    """Spatial attention sub-module for CBAM."""

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            1, kernel_size=self.kernel_size, padding="same", use_bias=False
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = tf.nn.sigmoid(self.conv(concat))
        return inputs * attention


@keras.saving.register_keras_serializable(package="src")
class CBAM(layers.Layer):
    """Convolutional Block Attention Module."""

    def __init__(self, ratio: int = 8, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def call(self, inputs, training=None):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x


@keras.saving.register_keras_serializable(package="src")
class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block."""

    def __init__(self, ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = keras.Sequential(
            [
                layers.Dense(channels // self.ratio, activation="relu"),
                layers.Dense(channels, activation="sigmoid"),
            ]
        )
        super().build(input_shape)

    def call(self, inputs):
        se = self.squeeze(inputs)
        se = self.excitation(se)
        se = tf.reshape(se, (-1, 1, 1, inputs.shape[-1]))
        return inputs * se


def add_attention(x: tf.Tensor, attention_type: str = "cbam") -> tf.Tensor:
    """Apply attention block to feature maps."""
    if attention_type == "cbam":
        return CBAM()(x)
    if attention_type == "se":
        return SEBlock()(x)
    return x
