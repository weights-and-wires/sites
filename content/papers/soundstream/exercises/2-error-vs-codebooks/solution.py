import numpy as np


def kmeans(X, k, iters=15, seed=0):
    """Finished for you. Farthest-point initialization, then Lloyd iterations."""
    rng = np.random.default_rng(seed)
    c = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min([((X - ci) ** 2).sum(-1) for ci in c], axis=0)
        c.append(X[d.argmax()])
    c = np.array(c, dtype=float)
    for _ in range(iters):
        a = ((X[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if np.any(a == j):
                c[j] = X[a == j].mean(0)
    return c


def quantize(codebook, R):
    """Finished for you. Replace each row of R by its nearest code vector."""
    return codebook[((R[:, None, :] - codebook[None]) ** 2).sum(-1).argmin(1)]


def fit_rvq(X, n_q, n_codes, seed=0):
    """Return a list of n_q codebooks, each of shape (n_codes, D).

    Stage i is fitted with kmeans(residual, n_codes, seed=seed + i), where
    residual is what stages 0..i-1 left behind.
    """
    residual = np.array(X, dtype=float)
    books = []
    for i in range(n_q):
        c = kmeans(residual, n_codes, seed=seed + i)
        books.append(c)
        residual = residual - quantize(c, residual)
    return books


def mse_by_nq(X, codebooks):
    """Return an array of length len(codebooks) + 1.

    Entry n is np.mean((X - X_hat) ** 2) when only the first n codebooks are
    used. Entry 0 uses X_hat = 0.
    """
    X = np.asarray(X, dtype=float)
    x_hat = np.zeros_like(X)
    out = [float(np.mean(X ** 2))]
    for c in codebooks:
        x_hat = x_hat + quantize(c, X - x_hat)
        out.append(float(np.mean((X - x_hat) ** 2)))
    return np.array(out)
