"""
MobileNetV3-Small + CBAM ICDAS ordinal classifier.
"""

from __future__ import annotations

import keras
from keras import applications, layers

from .attention import add_attention
from .icdas import NUM_CLASSES
from .losses import ordinal_predict as ordinal_to_class


def build_model(
    num_classes: int = NUM_CLASSES,
    image_size: int = 224,
    attention_type: str = "cbam",
    dropout: float = 0.3,
    pretrained: bool = True,
    ordinal_regression: bool = True,
    ordinal: bool | None = None,
) -> keras.Model:
    if ordinal is not None:
        ordinal_regression = ordinal

    inputs = keras.Input(
        shape=(image_size, image_size, 3),
        name="image",
    )

    weights = "imagenet" if pretrained else None

    backbone = applications.MobileNetV3Small(
        input_tensor=inputs,
        include_top=False,
        weights=weights,
        pooling=None,
    )

    # Stage 1 default: freeze backbone for head training.
    backbone.trainable = False

    x = backbone.output

    if attention_type and attention_type != "none":
        x = add_attention(x, attention_type)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="fc_hidden")(x)
    x = layers.Dropout(dropout * 0.5, name="dropout_2")(x)

    # Ordinal regression:
    #   K classes -> K-1 thresholds
    #   ICDAS 0–4 => 5 classes => 4 sigmoid outputs
    #   output[k] = P(y > k), k = 0,1,2,3
    if ordinal_regression:
        outputs = layers.Dense(
            num_classes - 1,
            activation="sigmoid",
            dtype="float32",
            name="ordinal",
        )(x)
    else:
        outputs = layers.Dense(
            num_classes,
            activation="softmax",
            dtype="float32",
            name="class",
        )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="icdas_mobilenet_cbam",
    )


def unfreeze_top_layers(model: keras.Model, num_layers: int = 30) -> None:
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        print("WARNING: MobileNet backbone not found.")
        return

    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - num_layers)
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= freeze_until

    print(f"Unfroze top {num_layers} MobileNet layers.")


def get_custom_objects():
    from .attention import CBAM, ChannelAttention, SEBlock, SpatialAttention

    return {
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "SEBlock": SEBlock,
    }


def get_last_conv_layer(model: keras.Model):
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, layers.Conv2D):
                    return sublayer
    raise ValueError("No Conv2D layer found.")


__all__ = [
    "build_model",
    "unfreeze_top_layers",
    "get_custom_objects",
    "get_last_conv_layer",
    "ordinal_to_class",
]
