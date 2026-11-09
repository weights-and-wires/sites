"""Compute the numbers for the additive-attention example.

Run: python3 scripts/additive_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Notation follows Appendix A.1.2: e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j).
# Sizes: n = 1 (decoder state), 2n = 2 (annotation), n' = 1 (alignment hidden units).
s_prev = np.array([0.0])
H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # h_1, h_2, h_3
Wa = np.array([[1.0]])
Ua = np.array([[1.0, -1.0]])
va = np.array([2.0])

pre = Wa @ s_prev + H @ Ua.T  # (3, 1): U_a h_j, since W_a s = 0
e = np.tanh(pre) @ va
alpha = np.exp(e - e.max()) / np.exp(e - e.max()).sum()
c = alpha @ H

print("W_a s_{i-1} + U_a h_j:", pre[:, 0])
print("e_ij:", e)
print("alpha_ij:", alpha, "sum =", alpha.sum())
print("c_i:", c)
print("check, hand sum alpha_1 h_1 + alpha_2 h_2 + alpha_3 h_3:", alpha[0] * H[0] + alpha[1] * H[1] + alpha[2] * H[2])

# Second state: s = [1]. The ordering of scores can change only if tanh is nonlinear.
s2 = np.array([1.0])
e2 = np.tanh(Wa @ s2 + H @ Ua.T) @ va
a2 = np.exp(e2 - e2.max()) / np.exp(e2 - e2.max()).sum()
print("with s = 1: e =", e2, " alpha =", a2)

# Self-check item: scores [ln 3, 0, 0] give weights [0.6, 0.2, 0.2], and the same H gives c = [0.8, 0.4].
e3 = np.array([np.log(3.0), 0.0, 0.0])
a3 = np.exp(e3) / np.exp(e3).sum()
print("self-check weights:", a3, " context:", a3 @ H)
