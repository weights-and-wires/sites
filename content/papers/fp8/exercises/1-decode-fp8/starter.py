import math

FORMATS = {
    "e4m3": {"ebits": 4, "mbits": 3, "bias": 7},
    "e5m2": {"ebits": 5, "mbits": 2, "bias": 15},
}


def decode(code, fmt):
    """Return the float encoded by the 8-bit pattern `code` (0..255) in `fmt`.

    Layout is S.E.M: 1 sign bit, then `ebits` exponent bits, then `mbits`
    mantissa bits. Return float("nan") and +/- float("inf") for special values.
    """
    raise NotImplementedError
