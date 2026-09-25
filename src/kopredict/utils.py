"""Utility functions."""

import time
from math import floor, log10


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


def time_function(func, *args, **kwargs):
    """Run func(*args, **kwargs) and return (result, elapsed_seconds)."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start
