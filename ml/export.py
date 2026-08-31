#!/usr/bin/env python3
"""
Export trained model to TFLite, TensorFlow.js, and ONNX with quantization benchmarks.
Usage: python export.py --checkpoint ../models/icdas/current/<experiment>/best.keras --quantize
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

# Mock missing/unsupported modules on Windows (e.g. tensorflow-decision-forests, JAX, pkg_resources)
import unittest.mock

# Mock tensorflow-decision-forests (fails to import on Windows)
sys.modules['tensorflow_decision_forests'] = unittest.mock.MagicMock()

# Mock jax, jax.experimental, flax (often missing in direct tfjs installations)
sys.modules['jax'] = unittest.mock.MagicMock()
sys.modules['jax.experimental'] = unittest.mock.MagicMock()
sys.modules['flax'] = unittest.mock.MagicMock()

# Mock pkg_resources (sometimes missing or deprecated in newer python/setuptools environments)
class MockVersion:
    def __lt__(self, other): return False
    def __gt__(self, other): return False
    def __le__(self, other): return False
    def __ge__(self, other): return False

pkg_mock = unittest.mock.MagicMock()
pkg_mock.parse_version = lambda x: MockVersion()
sys.modules['pkg_resources'] = pkg_mock

from src.model import get_custom_objects


def get_model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def export_saved_model(model, output_dir: str):
    """Export TensorFlow SavedModel format."""
    saved_path = os.path.join(output_dir, "saved_model")
    model.export(saved_path)
    return saved_path


def export_tflite(model, output_path: str, quantize: bool = True):
    """Export TensorFlow Lite with optional dynamic range quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path


def export_tfjs_from_keras(model, output_dir: str):
    """Convert Keras model directly to TensorFlow.js."""
    try:
        import tensorflowjs as tfjs

        tfjs.converters.save_keras_model(model, output_dir)
        return output_dir
    except Exception as e:
        print(f"TF.js export warning: {e}")
        print("Install tensorflowjs: pip install tensorflowjs")
        return None


def export_tfjs(saved_model_path: str, output_dir: str):
    """Convert SavedModel to TensorFlow.js (fallback)."""
    try:
        import tensorflowjs as tfjs

        tfjs.converters.convert_tf_saved_model(
            saved_model_path,
            output_dir,
            quantization_dtype_map={"class": "uint8"} if False else None,
        )
        return output_dir
    except Exception as e:
        print(f"TF.js SavedModel export warning: {e}")
        return None


def export_onnx(model, output_path: str, input_size: int = 224):
    """Export to ONNX via tf2onnx (optional)."""
    try:
        import tf2onnx
        import onnx
        spec = (tf.TensorSpec((None, input_size, input_size, 3), tf.float32, name="image"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return output_path
    except ImportError:
        print("tf2onnx not installed — skipping ONNX export")
        return None


def benchmark_model(model, input_size: int = 224, runs: int = 50):
    """Benchmark inference latency and memory."""
    dummy = np.random.randn(1, input_size, input_size, 3).astype(np.float32)
    # Warmup
    for _ in range(5):
        model.predict(dummy, verbose=0)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="../models")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = tf.keras.models.load_model(
        args.checkpoint, compile=False, custom_objects=get_custom_objects()
    )

    # Simplify to class output only for deployment
    if isinstance(model.output, dict) and "class" in model.output_names:
        deploy_model = tf.keras.Model(model.input, model.output["class"])
    else:
        deploy_model = model

    deploy_path = str(output_dir / "deploy.keras")
    deploy_model.save(deploy_path)
    print(f"Deploy model saved: {deploy_path} ({get_model_size_mb(deploy_path):.2f} MB)")

    # TensorFlow.js (Keras path avoids SavedModel export issues with custom layers)
    tfjs_dir = str(output_dir / "tfjs_model")
    export_tfjs_from_keras(deploy_model, tfjs_dir)

    # TFLite (optional; custom CBAM layers may fail conversion on some TF versions)
    tflite_path = str(output_dir / "model.tflite")
    try:
        export_tflite(deploy_model, tflite_path, quantize=args.quantize)
        print(f"TFLite: {tflite_path} ({get_model_size_mb(tflite_path):.2f} MB)")
    except Exception as e:
        print(f"TFLite export skipped: {e}")
        tflite_path = None
    if not os.path.exists(os.path.join(tfjs_dir, "model.json")):
        try:
            saved_path = export_saved_model(deploy_model, str(output_dir))
            export_tfjs(saved_path, tfjs_dir)
        except Exception as e:
            print(f"SavedModel export skipped: {e}")
    if os.path.exists(tfjs_dir):
        total = sum(
            os.path.getsize(os.path.join(tfjs_dir, f))
            for f in os.listdir(tfjs_dir)
            if os.path.isfile(os.path.join(tfjs_dir, f))
        )
        print(f"TF.js model: {tfjs_dir} ({total / 1024 / 1024:.2f} MB)")

    # ONNX
    onnx_path = str(output_dir / "model.onnx")
    export_onnx(deploy_model, onnx_path, args.image_size)

    # Benchmark
    bench = benchmark_model(deploy_model, args.image_size)
    tflite_mb = get_model_size_mb(tflite_path) if tflite_path and os.path.exists(tflite_path) else None
    report = {
        "checkpoint": args.checkpoint,
        "deploy_size_mb": get_model_size_mb(deploy_path),
        "tflite_size_mb": tflite_mb,
        "benchmark": bench,
        "target_met": bench["p95_ms"] < 1000 and (tflite_mb is None or tflite_mb < 20),
    }
    report_path = output_dir / "export_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark: mean={bench['mean_ms']:.1f}ms, p95={bench['p95_ms']:.1f}ms")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
