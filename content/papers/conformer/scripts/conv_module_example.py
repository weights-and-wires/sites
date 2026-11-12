#!/usr/bin/env python3
"""Numbers for the convolution module idea (Section 2.2, Figure 2)."""

import math

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def depthwise_conv1d(x, w):
    T, d = x.shape
    k = w.shape[0]
    left = (k - 1) // 2
    xp = np.pad(x, ((left, k - 1 - left), (0, 0)))
    out = np.zeros((T, d))
    for i in range(k):
        out += w[i] * xp[i:i + T]
    return out


def naive(x, w):
    T, d = x.shape
    k = w.shape[0]
    left = (k - 1) // 2
    y = np.zeros((T, d))
    for t in range(T):
        for c in range(d):
            for i in range(k):
                j = t + i - left
                if 0 <= j < T:
                    y[t, c] += w[i, c] * x[j, c]
    return y


# Depthwise convolution on one channel.
x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
w = np.array([[1.0], [0.0], [-1.0]])
print("depthwise, k=3, w=[1, 0, -1]:", depthwise_conv1d(x, w)[:, 0])
imp = np.zeros((7, 1))
imp[3, 0] = 1.0
print("impulse at frame 3, w=[1, 2, 3]:", depthwise_conv1d(imp, np.array([[1.0], [2.0], [3.0]]))[:, 0])

# Depthwise against the naive loop, kernel 32 (Table 1).
rng = np.random.default_rng(0)
xr = rng.normal(size=(100, 8))
wr = rng.normal(size=(32, 8))
print("k=32 max |vectorized - naive|: %.2e" % np.abs(depthwise_conv1d(xr, wr) - naive(xr, wr)).max())

# Parameter counts of one convolution module.
for d in (144, 256, 512):
    full = 32 * d * d  # a full convolution, d -> d, kernel 32
    dw = 32 * d
    print(f"d={d}: full conv {full}, depthwise {dw}, ratio {full // dw}")

# GLU and Swish values.
print("GLU(a=1, b=0) = %.4f" % (1 * sigmoid(0.0)))
print("GLU(a=1, b=2) = %.4f" % (1 * sigmoid(2.0)))
print("GLU(a=1, b=-4) = %.4f" % (1 * sigmoid(-4.0)))
for v in (-2.0, 0.0, 1.0, 3.0):
    print("swish(%.0f) = %.4f   relu = %.1f" % (v, v * sigmoid(v), max(v, 0.0)))

# Batch norm over time on one channel.
z = np.array([1.0, 2.0, 3.0, 6.0])
bn = (z - z.mean()) / np.sqrt(z.var() + 1e-5)
print("batch norm of [1, 2, 3, 6]: mean %.4f, var %.4f" % (z.mean(), z.var()), np.round(bn, 4))

# Shapes through the module, d=4, T=10.
T, d = 10, 4
print("pointwise 1:", (T, d), "->", (T, 2 * d), "| GLU ->", (T, d), "| depthwise ->", (T, d), "| pointwise 2 ->", (T, d))
