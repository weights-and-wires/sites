"""Build the joint sequence of Equation 6 on a toy example.

Run: python3 scripts/joint_sequence.py
"""

import numpy as np

Q, T, tau = 3, 4, 1
K = 2 * Q + 1
W = np.array([101, 102, 103, 104])  # toy text stream
A = np.arange(1, Q * T + 1).reshape(Q, T)  # Moshi audio, 1..12
B = 200 + np.arange(1, Q * T + 1).reshape(Q, T)  # user audio

def delayed(X):
    out = np.zeros_like(X)
    out[0] = X[0]
    out[1:, tau:] = X[1:, : T - tau]
    return out

V = np.vstack([W[None], delayed(A), delayed(B)])
print("K =", V.shape[0], "(expected", K, ")")
print(V)
print("rows sampled for Moshi (1-based):", list(range(1, Q + 2)))
print("rows fed from the user (1-based):", list(range(Q + 2, K + 1)))
