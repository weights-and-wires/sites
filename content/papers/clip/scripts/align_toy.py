#!/usr/bin/env python3
"""Align two toy modalities with linear encoders, as in Exercise 3.

Both views are linear images of the same 3-D hidden vectors. Gradient descent on
the symmetric contrastive loss should make every image retrieve its own text.

    python3 scripts/align_toy.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "exercises/3-align-toy-modalities"))
from solution import loss_and_grads  # noqa: E402

rng = np.random.default_rng(5)
n = 8
z = rng.normal(size=(n, 3))
Xi = z @ rng.normal(size=(3, 6)) + 0.05 * rng.normal(size=(n, 6))
Xt = z @ rng.normal(size=(3, 5)) + 0.05 * rng.normal(size=(n, 5))
Wi = 0.3 * rng.normal(size=(6, 4))
Wt = 0.3 * rng.normal(size=(5, 4))
t = np.log(1 / 0.07)


def retrieval(Wi, Wt):
    Ie, Te = Xi @ Wi, Xt @ Wt
    Ie /= np.linalg.norm(Ie, axis=1, keepdims=True)
    Te /= np.linalg.norm(Te, axis=1, keepdims=True)
    S = Ie @ Te.T
    return (S.argmax(axis=1) == np.arange(n)).mean(), (S.argmax(axis=0) == np.arange(n)).mean()


print(f"n = {n}, ln n = {np.log(n):.4f}, scale = {np.exp(t):.4f}")
for step in range(1501):
    loss, dWi, dWt, _ = loss_and_grads(Xi, Xt, Wi, Wt, t)
    if step in (0, 100, 500, 1500):
        ai, at = retrieval(Wi, Wt)
        print(f"step {step:>4}: loss {loss:.4f}, image-to-text accuracy {ai:.3f}, text-to-image accuracy {at:.3f}")
    Wi = Wi - 1.0 * dWi
    Wt = Wt - 1.0 * dWt
