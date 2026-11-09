"""Fit a frozen random layer to a rank-2 target with LoRA at several ranks.

This is the setup of exercise 4. It imports the exercise solution so that the
page and the exercise use the same code.

Run: python3 scripts/fit_frozen_layer.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "exercises/4-fit-frozen-layer"))
from solution import train_lora, trainable_params  # noqa: E402

d, k, n = 12, 12, 200
rng = np.random.default_rng(0)
W0 = rng.normal(size=(d, k)) / np.sqrt(k)
U = rng.normal(size=(d, 2))
V = rng.normal(size=(2, k))
DELTA = 0.5 * (U @ V) / np.sqrt(k)
X = rng.normal(size=(n, k))
Y = X @ (W0 + DELTA).T

print("rank of the target update =", np.linalg.matrix_rank(DELTA))
print("singular values of the target update =", np.round(np.linalg.svd(DELTA, compute_uv=False)[:3], 4).tolist())
print("full update would train   =", d * k, "weights")
for r in (1, 2, 4):
    _, _, losses = train_lora(X, Y, W0, r=r, alpha=float(r), steps=1500, lr=0.2)
    print(f"r={r}: {trainable_params(d, k, r):3d} trainable ({100 * trainable_params(d, k, r) / (d * k):.0f}% of {d * k}); loss {losses[0]:.4f} -> {losses[-1]:.6f}")
