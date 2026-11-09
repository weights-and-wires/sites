"""Compute receptive fields of causal convolution stacks, by formula and by brute force.

Run: python3 scripts/receptive_field.py
"""

import numpy as np


def causal_conv(x, w, dilation):
    """y[t] = sum_j w[j] * x[t - (k-1-j)*dilation], with zeros before t = 0."""
    k = len(w)
    y = np.zeros_like(x, dtype=float)
    for j in range(k):
        shift = (k - 1 - j) * dilation
        if shift == 0:
            y += w[j] * x
        else:
            y[shift:] += w[j] * x[:-shift]
    return y


def formula(dilations, k=2):
    return 1 + (k - 1) * sum(dilations)


def measured(dilations, k=2):
    """Send a unit impulse through an all-ones linear stack; find the farthest output it reaches."""
    n = formula(dilations, k) + 50
    x = np.zeros(n)
    x[10] = 1.0
    for d in dilations:
        x = causal_conv(x, np.ones(k), d)
    return int(np.nonzero(x)[0].max() - 10) + 1


block = [2**i for i in range(10)]
cases = [
    ("4 undilated layers, k=2 (Figure 2)", [1, 1, 1, 1], 2),
    ("dilations 1, 2, 4, 8 (Figure 3)", [1, 2, 4, 8], 2),
    ("one block 1, 2, ..., 512", block, 2),
    ("three blocks", block * 3, 2),
    ("one block 1, 2, ..., 512 with k=3", block, 3),
]
for name, dil, k in cases:
    f, m = formula(dil, k), measured(dil, k)
    print(f"{name}: layers={len(dil)} formula={f} measured={m} at 16 kHz={1000 * f / 16000:.1f} ms")
