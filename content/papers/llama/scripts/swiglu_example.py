"""Compute the numbers for the SwiGLU example.

Run: python3 scripts/swiglu_example.py
"""

import math

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def silu(a):
    return a / (1.0 + np.exp(-a))


# d = 2, hidden width 3.
x = np.array([1.0, 2.0])
W1 = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])  # gate branch ("W" in Shazeer)
V = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])  # value branch
W2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
a, b = x @ W1, x @ V
print("x W1 =", a.tolist())
print("x V  =", b.tolist())
print("silu(x W1) =", silu(a).tolist())
print("gated      =", (silu(a) * b).tolist())
print("output     =", (silu(a) * b @ W2).tolist())
print("swap roles (silu on the V branch):", ((silu(b) * a) @ W2).tolist())
print("silu(-10) =", float(silu(np.array(-10.0))), "; silu(0) =", float(silu(np.array(0.0))), "; silu(10) =", round(float(silu(np.array(10.0))), 4))

# Parameter counts with d = 4096 (LLaMA-7B).
d = 4096
mlp = 2 * d * (4 * d)
print()
print("GELU/ReLU MLP, hidden 4d =", 4 * d, ": weights =", mlp, "(= 8 d^2 =", 8 * d * d, ")")
print("SwiGLU with hidden 4d (3 matrices):", 3 * d * 4 * d, "=", round(3 * d * 4 * d / mlp, 2), "x the MLP")
h_exact = 2 * 4 * d / 3
print("2/3 * 4d =", round(h_exact, 2))
h_floor = int(h_exact)
print("SwiGLU with hidden", h_floor, ":", 3 * d * h_floor, "weights, ratio", round(3 * d * h_floor / mlp, 6))


def llama_hidden(d, multiple_of=256):
    return multiple_of * math.ceil(int(2 * 4 * d / 3) / multiple_of)


h = llama_hidden(d)
print("rounded up to a multiple of 256:", h, ":", 3 * d * h, "weights, ratio", round(3 * d * h / mlp, 4))
