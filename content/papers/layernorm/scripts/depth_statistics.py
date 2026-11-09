"""Activation statistics through a 20-layer random ReLU MLP, with and without layer norm.

Layer l computes a = W h, optionally a = LN(a) (gain 1, bias 0), then h = relu(a).
W has entries from N(0, s^2 / H). For ReLU, s = sqrt(2) keeps the second moment
constant, so s = 2 grows it and s = 1 shrinks it. We record the standard deviation
of a over all examples and features at each layer.

Run: python3 scripts/depth_statistics.py
"""

import numpy as np


def ln(a, eps=1e-5):
    mu = a.mean(-1, keepdims=True)
    var = a.var(-1, keepdims=True)
    return (a - mu) / np.sqrt(var + eps)


def run(scale, use_ln, layers=20, n=64, h=256, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, h))
    stds = []
    for _ in range(layers):
        w = rng.normal(size=(h, h)) * scale / np.sqrt(h)
        a = x @ w.T
        if use_ln:
            a = ln(a)
        stds.append(float(a.std()))
        x = np.maximum(a, 0)
    return stds


for scale in (2.0, 1.0):
    print(f"weight scale s = {scale}")
    for use_ln in (False, True):
        s = run(scale, use_ln)
        tag = "with LN   " if use_ln else "without LN"
        print(f"  {tag} std of a at layers 1, 5, 10, 20: " + ", ".join(f"{s[i - 1]:.3g}" for i in (1, 5, 10, 20)))
