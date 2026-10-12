"""Utility functions."""

from math import floor, log10

import numpy as np


def try_float_conversion(s):
    """Try to convert a string to float; return original if conversion fails."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


def sig_figs(x: float, precision: int):
    """Round a number to the given number of significant figures."""
    x = float(x)
    precision = int(precision)
    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))


def hash_feature_rows(encoded: np.ndarray) -> np.ndarray:
    """Hash each encoded feature row so identical rows share one bucket.

    ``encoded`` is a uint8 matrix. Callers still compare rows inside a bucket;
    the hash only narrows that search.
    """
    data = np.ascontiguousarray(encoded)
    pad = (-data.shape[1]) % 8
    if pad:
        data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
    folded = data.view(np.uint64)
    hashes = np.zeros(data.shape[0], dtype=np.uint64)
    for column in range(folded.shape[1]):
        hashes ^= folded[:, column] * np.uint64(0x9E3779B97F4A7C15 + column)
    return hashes
