"""Compute the rank and low-rank approximation numbers for the rank example.

Run: python3 scripts/rank_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# A delta W with two singular values, 3 and 1.
U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
Vt = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
S = np.diag([3.0, 1.0])
dW = U @ S @ Vt
print("Delta W =", dW.tolist())
print("rank    =", np.linalg.matrix_rank(dW))
print("singular values =", np.linalg.svd(dW, compute_uv=False).tolist())

# The best rank-1 update keeps the top singular direction.
u, sv, vt = np.linalg.svd(dW)
best1 = sv[0] * np.outer(u[:, 0], vt[0])
print("best rank-1 =", best1.tolist())
print("Frobenius error of rank 1 =", np.linalg.norm(dW - best1))
print("Frobenius norm of Delta W =", np.linalg.norm(dW))
print("Frobenius error of rank 2 =", np.linalg.norm(dW - dW))

# Any product B A with r columns has rank at most r.
rng = np.random.default_rng(0)
for r in (1, 2, 3):
    BA = rng.normal(size=(6, r)) @ rng.normal(size=(r, 6))
    print(f"random 6x6 B A with r={r}: rank {np.linalg.matrix_rank(BA)}")

# Where the factorization of the best rank-1 update comes from.
B = (np.sqrt(sv[0]) * u[:, :1])
A = (np.sqrt(sv[0]) * vt[:1])
print("B shape", B.shape, "A shape", A.shape, "BA equals best rank-1:", bool(np.allclose(B @ A, best1)))
