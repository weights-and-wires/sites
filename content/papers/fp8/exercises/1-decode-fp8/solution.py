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
    f = FORMATS[fmt]
    eb, mb, bias = f["ebits"], f["mbits"], f["bias"]
    sign = -1.0 if (code >> 7) & 1 else 1.0
    e = (code >> mb) & ((1 << eb) - 1)
    m = code & ((1 << mb) - 1)
    if fmt == "e4m3" and e == (1 << eb) - 1 and m == (1 << mb) - 1:
        return float("nan")
    if fmt == "e5m2" and e == (1 << eb) - 1:
        return sign * float("inf") if m == 0 else float("nan")
    if e == 0:
        return sign * (m / 2**mb) * 2.0 ** (1 - bias)
    return sign * (1 + m / 2**mb) * 2.0 ** (e - bias)
