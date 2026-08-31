"""Average-hash perceptual fingerprint for training-image duplicate flags."""

from __future__ import annotations

import numpy as np

LIKELY_HAMMING = 8


def average_hash_hex(rgb: np.ndarray, size: int = 8) -> str:
    """64-bit average hash as 16 hex chars. Does not decode ICDAS."""
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    im = Image.fromarray(arr.astype(np.uint8)).convert("L")
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(im, dtype=np.float32)
    mean = float(pixels.mean())
    bits = (pixels >= mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    width = size * size // 4
    return f"{value:0{width}x}"


def hamming_hex(a: str | None, b: str | None) -> int:
    if not a or not b:
        return 64
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


def is_likely_duplicate(phash_a: str | None, phash_b: str | None, threshold: int = LIKELY_HAMMING) -> bool:
    return hamming_hex(phash_a, phash_b) <= threshold
