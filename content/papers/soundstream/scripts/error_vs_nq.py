"""Train a residual quantizer with k-means per stage and print MSE against nq.

Run: python3 scripts/error_vs_nq.py
"""

import numpy as np

rng = np.random.default_rng(0)
D, N, N_Q, n_points = 8, 16, 8, 4000
X = rng.normal(size=(n_points, D))


def kmeans(X, k, iters=15, seed=0):
    r = np.random.default_rng(seed)
    c = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(k):
            if np.any(a == j):
                c[j] = X[a == j].mean(0)
    return c


def quantize(c, R):
    d = ((R[:, None, :] - c[None]) ** 2).sum(-1)
    return c[d.argmin(1)]


books = []
R = X.copy()
mse = [float((X ** 2).mean())]
for i in range(N_Q):
    c = kmeans(R, N, seed=i)
    books.append(c)
    R = R - quantize(c, R)
    mse.append(float((R ** 2).mean()))

print("nq  bits/frame  MSE     ratio to previous")
for nq, m in enumerate(mse):
    ratio = "" if nq == 0 else f"{m / mse[nq - 1]:.3f}"
    print(f"{nq:2d}  {nq * 4:10d}  {m:.4f}  {ratio}")
print(f"monotone decreasing: {all(a > b for a, b in zip(mse, mse[1:]))}")

print("\nBug: fit every stage on X instead of the residual")
R = X.copy()
y_hat = np.zeros_like(X)
bug = []
for i in range(N_Q):
    c = kmeans(X, N, seed=i)
    y_hat = y_hat + quantize(c, R)
    R = X - y_hat
    bug.append(float((R ** 2).mean()))
print("MSE by nq:", [round(m, 4) for m in bug])
print(f"correct MSE at nq = {N_Q}: {mse[-1]:.4f}; buggy: {bug[-1]:.4f}")
