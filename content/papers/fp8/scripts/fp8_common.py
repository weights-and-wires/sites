"""Shared FP8 helpers for the scripts on this page (Section 3, Table 1)."""
import numpy as np

# name: (exponent bits, mantissa bits, bias, max normal, special-value rule)
FORMATS = {
    "e4m3": {"ebits": 4, "mbits": 3, "bias": 7, "max": 448.0},
    "e5m2": {"ebits": 5, "mbits": 2, "bias": 15, "max": 57344.0},
}


def decode(code, fmt):
    """Decode an 8-bit pattern S.E.M into a float (Table 1)."""
    f = FORMATS[fmt]
    eb, mb, bias = f["ebits"], f["mbits"], f["bias"]
    sign = -1.0 if (code >> 7) & 1 else 1.0
    e = (code >> mb) & ((1 << eb) - 1)
    m = code & ((1 << mb) - 1)
    if fmt == "e4m3" and e == 15 and m == 7:
        return float("nan")
    if fmt == "e5m2" and e == 31:
        return sign * float("inf") if m == 0 else float("nan")
    if e == 0:
        return sign * (m / 2**mb) * 2.0 ** (1 - bias)
    return sign * (1 + m / 2**mb) * 2.0 ** (e - bias)


def grid(fmt):
    """Sorted array of all finite non-negative values."""
    vals = [decode(c, fmt) for c in range(128)]
    return np.array(sorted(v for v in vals if np.isfinite(v)))


def quantize(x, fmt):
    """Round to nearest even in the FP8 format and saturate on overflow."""
    f = FORMATS[fmt]
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    _, e = np.frexp(np.where(a > 0, a, 1.0))
    e = np.maximum(e - 1, 1 - f["bias"])
    spacing = 2.0 ** (e - f["mbits"])
    q = np.minimum(np.round(a / spacing) * spacing, f["max"])
    return np.where(np.isnan(x), np.nan, np.sign(x) * q)
