import numpy as np


def fit_pca(X, k):
    """Return (mean, W): the data mean (D,) and the top-k principal directions (k, D)."""
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt[:k]


def encode(X, mean, W):
    """Return z = E(x), shape (n, k)."""
    return (X - mean) @ W.T


def decode(Z, mean, W):
    """Return the reconstruction D(z), shape (n, D)."""
    return Z @ W + mean


def analytic_eps(zt, abar, m, Sigma):
    """Return the exact noise prediction for Gaussian latents N(m, Sigma); shape (n, k)."""
    k = zt.shape[1]
    Sigma_t = abar * Sigma + (1.0 - abar) * np.eye(k)
    return np.sqrt(1.0 - abar) * np.linalg.solve(Sigma_t, (zt - np.sqrt(abar) * m).T).T


def ddpm_sample(eps_fn, num, k, betas, rng):
    """Finished for you: ancestral DDPM sampling with variance beta_t."""
    abars = np.cumprod(1.0 - betas)
    z = rng.standard_normal((num, k))
    for t in range(len(betas) - 1, -1, -1):
        z = (z - betas[t] / np.sqrt(1.0 - abars[t]) * eps_fn(z, abars[t])) / np.sqrt(1.0 - betas[t])
        if t > 0:
            z = z + np.sqrt(betas[t]) * rng.standard_normal(z.shape)
    return z


def generate(num, mean, W, m, Sigma, betas, seed=0):
    """Finished for you: sample latents with the analytic denoiser, then decode."""
    rng = np.random.default_rng(seed)
    k = W.shape[0]
    z = ddpm_sample(lambda zt, ab: analytic_eps(zt, ab, m, Sigma), num, k, betas, rng)
    return decode(z, mean, W)
