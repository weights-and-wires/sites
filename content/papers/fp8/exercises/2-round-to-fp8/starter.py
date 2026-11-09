import numpy as np

FORMATS = {
    "e4m3": {"mbits": 3, "bias": 7, "max": 448.0},
    "e5m2": {"mbits": 2, "bias": 15, "max": 57344.0},
}


def quantize(x, fmt):
    """Round `x` to the nearest FP8 value in `fmt`, saturating on overflow.

    Ties go to even. NaN stays NaN. Returns a float64 array shaped like `x`.
    """
    raise NotImplementedError
