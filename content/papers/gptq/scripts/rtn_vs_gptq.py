#!/usr/bin/env python3
"""Round-to-nearest versus GPTQ on one synthetic linear layer (Equation 1)."""

import numpy as np
from gptq_lib import correlated_inputs, gptq, layer_loss, rtn

d_row, d_col, m = 16, 32, 256

print("relative layer error ||WX - QX||^2 / ||WX||^2, mean over 20 seeds")
print("bits   RTN        GPTQ       GPTQ/RTN")
for bits in (4, 3, 2):
    r, g = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(d_row, d_col))
        X = correlated_inputs(rng, d_col, m)
        ref = float(np.sum((W @ X) ** 2))
        r.append(layer_loss(W, rtn(W, bits), X) / ref)
        g.append(layer_loss(W, gptq(W, X, bits), X) / ref)
    print(f"{bits}      {np.mean(r):.5f}    {np.mean(g):.5f}    {np.mean(g) / np.mean(r):.3f}")

rng = np.random.default_rng(0)
W = rng.normal(size=(d_row, d_col))
X = correlated_inputs(rng, d_col, m)
ref = float(np.sum((W @ X) ** 2))
Q_r, Q_g = rtn(W, 3), gptq(W, X, 3)
print()
print("seed 0, 3 bits")
print(f"RTN  relative error  {layer_loss(W, Q_r, X) / ref:.5f}")
print(f"GPTQ relative error  {layer_loss(W, Q_g, X) / ref:.5f}")
print(f"RTN  mean |W - Q|    {np.abs(W - Q_r).mean():.4f}")
print(f"GPTQ mean |W - Q|    {np.abs(W - Q_g).mean():.4f}")
