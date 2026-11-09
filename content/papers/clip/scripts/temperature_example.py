#!/usr/bin/env python3
"""The learnable temperature of CLIP: initial value, clip, and effect on the loss.

Section 2.5 initializes tau to the equivalent of 0.07 and clips the logit scale at 100.
Figure 3 stores the log of the scale in a parameter t, so scale = exp(t).

    python3 scripts/temperature_example.py
"""

import numpy as np

tau0 = 0.07
t0 = np.log(1 / tau0)
print(f"initial scale 1/tau = {1 / tau0:.4f}, t = ln(1/tau) = {t0:.4f}")
print(f"clip at scale 100:   t_max = ln(100) = {np.log(100):.4f}, tau_min = {1 / 100}")

# The same 4-pair batch at three scales: noisy embeddings of unit vectors.
rng = np.random.default_rng(0)
n, d = 4, 8
base = rng.normal(size=(n, d))
I = base + 0.5 * rng.normal(size=(n, d))
T = base + 0.5 * rng.normal(size=(n, d))
I /= np.linalg.norm(I, axis=1, keepdims=True)
T /= np.linalg.norm(T, axis=1, keepdims=True)
S = I @ T.T
print("\ncosine similarities of the 4-pair batch:")
print(np.round(S, 3))
print("diagonal:", np.round(np.diag(S), 3), " largest off-diagonal:", round(float(S[~np.eye(n, dtype=bool)].max()), 3))


def loss(logits):
    def lse(x, axis):
        m = x.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)
    d_ = np.diag(logits)
    return ((lse(logits, 1) - d_).mean() + (lse(logits, 0) - d_).mean()) / 2


print("\nscale   loss     mean max-prob over texts")
for s in (1.0, 1 / tau0, 100.0):
    lg = s * S
    z = np.exp(lg - lg.max(axis=1, keepdims=True))
    p = z / z.sum(axis=1, keepdims=True)
    print(f"{s:7.2f} {loss(lg):.5f}  {p.max(axis=1).mean():.5f}")
print(f"ln 4 = {np.log(4):.5f} is the loss at scale 0")
