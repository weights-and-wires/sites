"""Decoupled RoPE score (Section 2.1.3, Equations 14-18) on tiny inputs.

Run: python3 scripts/decoupled_rope_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    a = pos * theta
    out = np.empty_like(x, dtype=float)
    out[0::2] = x[0::2] * np.cos(a) - x[1::2] * np.sin(a)
    out[1::2] = x[0::2] * np.sin(a) + x[1::2] * np.cos(a)
    return out


d_h, d_hR = 2, 2
qC, kC = np.array([1.0, 2.0]), np.array([3.0, 1.0])
qR, kR = np.array([1.0, 0.0]), np.array([1.0, 0.0])
content = float(qC @ kC)
print("content score qC . kC       =", content)
for tq, tk in ((1, 0), (5, 4), (9, 8), (3, 0)):
    r = float(rope(qR, tq) @ rope(kR, tk))
    total = (content + r) / np.sqrt(d_h + d_hR)
    print(f"t={tq}, j={tk}: rope part = {r:.4f}, scaled score = {total:.4f}")
print("cos(1), cos(3)              =", np.cos(1.0), np.cos(3.0))
print("cache per token             =", "d_c + d_hR (latent plus one shared key)")
