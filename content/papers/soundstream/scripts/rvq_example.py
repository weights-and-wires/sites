"""Hand-checkable residual vector quantization in 2-D (Section III-C, Algorithm 1).

Run: python3 scripts/rvq_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

C1 = np.array([[0, 0], [4, 0], [0, 4], [4, 4]], dtype=float)
C2 = np.array([[0, 0], [-1, 1], [1, -1], [1, 1]], dtype=float)
C3 = 0.25 * C2
codebooks = [C1, C2, C3]
y = np.array([3.2, 0.9])


def nearest(c, v):
    d = ((c - v) ** 2).sum(-1)
    return int(np.argmin(d)), d


print("y =", y)
y_hat = np.zeros(2)
residual = y.copy()
for i, c in enumerate(codebooks, start=1):
    idx, d = nearest(c, residual)
    q = c[idx]
    y_hat = y_hat + q
    residual = residual - q
    print(f"stage {i}: distances {d.round(4).tolist()}, index {idx}, Q = {q.tolist()}")
    print(f"         y_hat = {y_hat.round(4).tolist()}, residual = {residual.round(4).tolist()}, |residual| = {np.linalg.norm(residual):.4f}")

print("\nStored vectors:", sum(len(c) for c in codebooks), "; reachable sums:", 4 ** 3)

print("\nBug: quantize y at every stage instead of the residual")
y_hat_bug = np.zeros(2)
for c in codebooks[:2]:
    idx, _ = nearest(c, y)
    y_hat_bug = y_hat_bug + c[idx]
print("y_hat after 2 stages =", y_hat_bug.tolist(), "; error =", round(float(np.linalg.norm(y - y_hat_bug)), 4))
print("correct y_hat after 2 stages has error", round(float(np.linalg.norm(residual + codebooks[2][2])), 4))
