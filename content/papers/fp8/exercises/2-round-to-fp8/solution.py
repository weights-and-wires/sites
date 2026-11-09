import numpy as np

FORMATS = {
    "e4m3": {"mbits": 3, "bias": 7, "max": 448.0},
    "e5m2": {"mbits": 2, "bias": 15, "max": 57344.0},
}


def quantize(x, fmt):
    """Round `x` to the nearest FP8 value in `fmt`, saturating on overflow.

    Ties go to even. NaN stays NaN. Returns a float64 array shaped like `x`.
    """
    f = FORMATS[fmt]
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    _, p = np.frexp(np.where(a > 0, a, 1.0))
    e = np.maximum(p - 1, 1 - f["bias"])
    spacing = 2.0 ** (e - f["mbits"])
    q = np.minimum(np.round(a / spacing) * spacing, f["max"])
    return np.where(np.isnan(x), np.nan, np.sign(x) * q)
