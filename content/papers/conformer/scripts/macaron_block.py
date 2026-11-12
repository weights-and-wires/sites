#!/usr/bin/env python3
"""Numbers for the Macaron block idea (Section 2.4, Equation 1)."""

import numpy as np


def block(x, ffn1, mhsa, conv, ffn2, ln, half=0.5):
    x1 = x + half * ffn1(x)
    x2 = x1 + mhsa(x1)
    x3 = x2 + conv(x2)
    return ln(x3 + half * ffn2(x3)), (x1, x2, x3)


ident = lambda v: v
x = np.array([[1.0]])
ffn1 = lambda v: 2 * v
mhsa = lambda v: v ** 2
conv = lambda v: -v / 2
ffn2 = lambda v: 4 * v
y, (x1, x2, x3) = block(x, ffn1, mhsa, conv, ffn2, ident)
print("x=1: x~=%.1f, x'=%.1f, x''=%.1f, y=%.1f" % (x1[0, 0], x2[0, 0], x3[0, 0], y[0, 0]))
y_full, _ = block(x, ffn1, mhsa, conv, ffn2, ident, half=1.0)
print("full-step residuals: y=%.1f" % y_full[0, 0])

# Final layer norm on a vector.
v = np.array([[1.0, 2.0, 3.0, 4.0]])
ln = (v - v.mean()) / np.sqrt(v.var() + 1e-5)
print("layer norm of [1, 2, 3, 4]:", np.round(ln[0], 4))
