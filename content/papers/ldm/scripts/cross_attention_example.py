#!/usr/bin/env python3
"""A tiny cross-attention layer in the paper's notation (Section 3.3).

phi_i(z_t) has N = 2 positions of width d_eps = 2.
tau_theta(y) has M = 3 tokens of width d_tau = 2.
The projection width is d = 2. Row-vector convention: Q = phi @ W_Q.T.
"""
import numpy as np

phi = np.array([[1.0, 0.0], [0.0, 1.0]])            # N x d_eps
tau = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # M x d_tau
W_Q = np.eye(2)
W_K = np.eye(2)
W_V = np.array([[1.0, 0.0], [0.0, 2.0]])

Q = phi @ W_Q.T
K = tau @ W_K.T
V = tau @ W_V.T
d = Q.shape[-1]
scores = Q @ K.T / np.sqrt(d)
e = np.exp(scores - scores.max(-1, keepdims=True))
w = e / e.sum(-1, keepdims=True)
out = w @ V

np.set_printoptions(precision=4, suppress=True)
print("scores (N x M):\n", scores)
print("weights (N x M):\n", w)
print("row sums:", w.sum(-1))
print("V (M x d):\n", V)
print("output (N x d):\n", out)

# Permuting the tokens changes nothing: cross-attention adds no word order.
perm = [2, 0, 1]
s2 = Q @ K[perm].T / np.sqrt(d)
e2 = np.exp(s2 - s2.max(-1, keepdims=True))
w2 = e2 / e2.sum(-1, keepdims=True)
print("output after permuting tokens:\n", w2 @ V[perm])
