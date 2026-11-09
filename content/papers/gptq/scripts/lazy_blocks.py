#!/usr/bin/env python3
"""Lazy batch updates (Step 2) change the schedule of the updates, not the result."""

import numpy as np
from gptq_lib import correlated_inputs, gptq, layer_loss

rng = np.random.default_rng(0)
W = rng.normal(size=(16, 32))
X = correlated_inputs(rng, 32, 256)

base = gptq(W, X, 3, blocksize=None)
print(f"unblocked loss {layer_loss(W, base, X):.4f}")
for B in (1, 4, 8, 16, 32):
    Q = gptq(W, X, 3, blocksize=B)
    print(f"B={B:<3} max |Q - Q_unblocked| = {np.abs(Q - base).max():.1e}   loss {layer_loss(W, Q, X):.4f}")
