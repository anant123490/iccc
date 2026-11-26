"""Unit tests for ML pipeline components."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.icdas import NUM_CLASSES, ORDINAL_THRESHOLDS, VALID_CLASSES
from src.losses import ordinal_loss, ordinal_predict, ordinal_to_class_probabilities  # noqa: E402
from src.model import build_model
from src.preprocessing import preprocess_image
from src.attention import CBAM
from src.dataset import discover_images_from_folders
from src.metrics import compute_metrics
from train import WholeValidationMetrics, get_callbacks


def test_build_model_softmax_production():
    model = build_model(
        num_classes=NUM_CLASSES,
        image_size=224,
        attention_type="cbam",
        ordinal_regression=False,
        pretrained=False,
    )
    assert model is not None
    x = np.random.randn(2, 224, 224, 3).astype(np.float32)
    out = model(x, training=False)
    tensor = out["class"] if isinstance(out, dict) else out
    assert tuple(tensor.shape) == (2, NUM_CLASSES)
    assert model.output_shape[-1] == 5


def test_build_model_default_is_softmax():
    model = build_model(pretrained=False)
    assert model.output_shape[-1] == NUM_CLASSES


def test_ordinal_head_still_available():
    model = build_model(
        num_classes=NUM_CLASSES,
        ordinal_regression=True,
        pretrained=False,
    )
    x = np.random.randn(1, 224, 224, 3).astype(np.float32)
    out = model(x, training=False)
    tensor = out["ordinal"] if isinstance(out, dict) else out
    assert tensor.shape[-1] == ORDINAL_THRESHOLDS
    assert model.output_shape[-1] == 4


def test_ordinal_decode_and_loss():
    # ICDAS 2 with 5 classes -> [1, 1, 0, 0]
    y_true = tf.constant([2], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.8, 0.1, 0.05]], dtype=tf.float32)
    loss = ordinal_loss(NUM_CLASSES)(y_true, y_pred)
    assert float(loss.numpy()) >= 0.0
    pred_class = int(ordinal_predict(y_pred).numpy()[0])
    assert pred_class == 2
    probs = ordinal_to_class_probabilities(y_pred.numpy())[0]
    assert probs.shape == (5,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert int(np.argmax(probs)) in VALID_CLASSES


def test_preprocess_mobilenet_range():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(
        img,
        target_size=224,
        use_roi=False,
        use_clahe=False,
        use_specular=False,
        color_norm=False,
    )
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 255.0
    assert result.max() > 1.5


def test_icdas_classes_are_zero_through_four():
    assert NUM_CLASSES == 5
    assert VALID_CLASSES == (0, 1, 2, 3, 4)


def test_cbam_layer():
    x = np.random.randn(1, 14, 14, 576).astype(np.float32)
    cbam = CBAM()
    out = cbam(x)
    assert out.shape == x.shape


def test_split_scoped_discovery_does_not_read_test_labels(tmp_path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "train.jpg").touch()
    (test_dir / "test.jpg").touch()
    (train_dir / "labels.csv").write_text(
        "filename,icdas_score\ntrain.jpg,0\n",
        encoding="utf-8",
    )
    # This invalid held-out label proves train-only discovery never opens it.
    (test_dir / "labels.csv").write_text(
        "filename,icdas_score\ntest.jpg,99\n",
        encoding="utf-8",
    )

    found = discover_images_from_folders(
        str(tmp_path),
        splits=("train",),
    )

    assert found["split"].tolist() == ["train"]
    assert found["icdas_score"].tolist() == [0]


def test_whole_validation_metrics_aggregate_all_batches():
    labels = tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)
    encoded_predictions = tf.constant([[0], [2], [1], [3], [4]], dtype=tf.int32)
    validation = tf.data.Dataset.from_tensor_slices(
        (encoded_predictions, labels)
    ).batch(2)

    class FixedPredictionModel:
        def __call__(self, images, training=False):
            assert training is False
            return tf.one_hot(tf.cast(images[:, 0], tf.int32), depth=5)

    callback = WholeValidationMetrics(validation, num_classes=5)
    callback.set_model(FixedPredictionModel())
    logs = {}
    callback.on_epoch_end(0, logs)

    expected = compute_metrics(
        labels.numpy(),
        np.array([0, 2, 1, 3, 4]),
        num_classes=5,
    )
    assert logs["val_qwk_global"] == pytest.approx(expected["quadratic_kappa"])
    assert logs["val_accuracy_global"] == pytest.approx(expected["accuracy"])
    assert logs["val_macro_f1_global"] == pytest.approx(expected["macro_f1"])


def test_stage_two_checkpoint_starts_from_stage_one_validation_best(tmp_path):
    validation = tf.data.Dataset.from_tensor_slices(
        (tf.zeros((5, 1)), tf.range(5, dtype=tf.int32))
    ).batch(5)
    config = {
        "dataset": {"num_classes": 5},
        "callbacks": {
            "monitor": "val_qwk_global",
            "mode": "max",
            "whole_validation_metrics": True,
            "checkpoint": {"enabled": True, "save_best_only": True},
            "early_stopping": {"enabled": True, "patience": 12},
            "reduce_lr": {"enabled": False},
        },
    }

    callbacks = get_callbacks(
        tmp_path,
        config,
        validation_dataset=validation,
        initial_best=0.31,
        append_history=True,
    )
    checkpoint = next(
        callback for callback in callbacks
        if isinstance(callback, tf.keras.callbacks.ModelCheckpoint)
    )
    assert isinstance(callbacks[0], WholeValidationMetrics)
    assert checkpoint.monitor == "val_qwk_global"
    assert checkpoint.best == pytest.approx(0.31)
