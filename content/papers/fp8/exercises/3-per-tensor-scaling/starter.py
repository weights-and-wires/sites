import numpy as np

FORMATS = {
    "e4m3": {"mbits": 3, "bias": 7, "max": 448.0},
    "e5m2": {"mbits": 2, "bias": 15, "max": 57344.0},
}


def quantize(x, fmt):
    """Finished for you: round to the nearest FP8 value, saturating on overflow."""
    f = FORMATS[fmt]
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    _, p = np.frexp(np.where(a > 0, a, 1.0))
    e = np.maximum(p - 1, 1 - f["bias"])
    spacing = 2.0 ** (e - f["mbits"])
    q = np.minimum(np.round(a / spacing) * spacing, f["max"])
    return np.where(np.isnan(x), np.nan, np.sign(x) * q)


def amax_scale(x, fmt):
    """Return fmt_max / max|x|, or 1.0 when x is all zeros."""
    raise NotImplementedError


def scaled_roundtrip(x, fmt):
    """Scale x, cast to FP8, and unscale. Returns a float64 array."""
    raise NotImplementedError
