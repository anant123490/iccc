"""Unit tests for ML pipeline components."""

import numpy as np
import pytest

# Skip TF tests if not installed
tf = pytest.importorskip("tensorflow")

from src.model import build_model, ordinal_to_class
from src.preprocessing import preprocess_image
from src.attention import CBAM, SEBlock


def test_build_model():
    model = build_model(num_classes=7, image_size=224, attention_type="cbam")
    assert model is not None
    x = np.random.randn(2, 224, 224, 3).astype(np.float32)
    out = model(x, training=False)
    assert "class" in out or isinstance(out, tf.Tensor)


def test_ordinal_head():
    model = build_model(num_classes=7, ordinal=True)
    x = np.random.randn(1, 224, 224, 3).astype(np.float32)
    out = model(x, training=False)
    if "ordinal" in out:
        assert out["ordinal"].shape[-1] == 6  # K-1 thresholds


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
