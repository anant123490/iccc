"""Unit tests for ML pipeline components."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.icdas import NUM_CLASSES, ORDINAL_THRESHOLDS
from src.losses import ordinal_loss, ordinal_predict, ordinal_to_class_probabilities  # noqa: E402
from src.model import build_model, ordinal_to_class
from src.preprocessing import preprocess_image
from src.attention import CBAM


def test_build_model():
    model = build_model(
        num_classes=NUM_CLASSES,
        image_size=224,
        attention_type="cbam",
        ordinal_regression=True,
    )
    assert model is not None
    x = np.random.randn(2, 224, 224, 3).astype(np.float32)
    out = model(x, training=False)
    tensor = out["ordinal"] if isinstance(out, dict) else out
    assert tuple(tensor.shape) == (2, ORDINAL_THRESHOLDS)


def test_ordinal_head():
    model = build_model(num_classes=NUM_CLASSES, ordinal=True)
    x = np.random.randn(1, 224, 224, 3).astype(np.float32)
    out = model(x, training=False)
    tensor = out["ordinal"] if isinstance(out, dict) else out
    assert tensor.shape[-1] == 4
    assert model.output_shape[-1] == 4


def test_ordinal_decode_and_loss():
    # ICDAS 2 with 5 classes -> [1, 1, 0, 0]
    y_true = tf.constant([2], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.8, 0.1, 0.05]], dtype=tf.float32)
    loss = ordinal_loss(NUM_CLASSES)(y_true, y_pred)
    assert float(loss.numpy()) >= 0.0
    pred_class = int(ordinal_predict(y_pred).numpy()[0])
    assert pred_class == 2
    assert int(ordinal_to_class(y_pred).numpy()[0]) == 2
    probs = ordinal_to_class_probabilities(y_pred.numpy())[0]
    assert probs.shape == (5,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert int(np.argmax(probs)) in range(NUM_CLASSES)


def test_preprocess():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(img, target_size=224, use_roi=False)
    assert result.shape == (224, 224, 3)
    assert result.max() <= 1.0


def test_cbam_layer():
    x = np.random.randn(1, 14, 14, 576).astype(np.float32)
    cbam = CBAM()
    out = cbam(x)
    assert out.shape == x.shape
