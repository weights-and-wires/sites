#!/usr/bin/env python3
"""Train a 4-expert Switch layer on the toy mixture task from exercise 4.

Four clusters of inputs each need their own linear map. A single linear map
(least squares) can't fit all four. The Switch layer uses the exercise
solution, so this script also checks that the page and the exercise agree.
"""

import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("solution", HERE.parent / "exercises/4-router-gradient/solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

rng = np.random.default_rng(2)
n = 400
centers = np.array([[3, 3], [-3, 3], [-3, -3], [3, -3.0]])
A = rng.normal(size=(4, 2, 2))
c = rng.integers(0, 4, n)
local = rng.normal(size=(n, 2))
x = np.concatenate([centers[c] + local, np.ones((n, 1))], axis=1)
y = np.einsum("nd,ndo->no", local, A[c])

B = np.linalg.lstsq(x, y, rcond=None)[0]
dense = float(np.mean((x @ B - y) ** 2))
print(f"target variance per output: {y.var():.4f}")
print(f"one linear map (least squares) MSE: {dense:.4f}")

_, cache = None, None
Wr, W, mse = sol.train(x, y, N=4)
_, cache = sol.switch_forward(x, Wr, W)
idx = cache[2]
print(f"4-expert Switch layer MSE after 2000 steps: {mse:.4f}")
print("tokens per expert:", np.bincount(idx, minlength=4).tolist())
table = np.zeros((4, 4), dtype=int)
for cluster, e in zip(c, idx):
    table[cluster, e] += 1
print("cluster (rows) by chosen expert (columns):")
print(table)
