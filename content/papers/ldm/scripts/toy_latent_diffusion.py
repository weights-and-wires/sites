#!/usr/bin/env python3
"""Toy latent diffusion: a PCA autoencoder plus an exact Gaussian denoiser.

Pixel space has D = 16 dimensions. The data lies close to a 2D subspace.
E and D are the PCA projection and its transpose. The "network" eps_theta is the
exact minimizer of Equation 2 for Gaussian latents, so no training is needed.
"""
import numpy as np

rng = np.random.default_rng(0)
D, k, n = 16, 2, 4000

basis, _ = np.linalg.qr(rng.standard_normal((D, k)))     # D x k
coef = rng.standard_normal((n, k)) * np.array([3.0, 1.0])
X = coef @ basis.T + 0.05 * rng.standard_normal((n, D)) + 2.0

mean = X.mean(0)
U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
W = Vt[:k]
Z = (X - mean) @ W.T
Xhat = Z @ W + mean
cov = np.cov(X - mean, rowvar=False, bias=True)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print("mean squared reconstruction error per sample:", round(float(((X - Xhat) ** 2).sum(1).mean()), 5))
print("sum of discarded eigenvalues:               ", round(float(eig[k:].sum()), 5))
print("latent variances:", np.round(Z.var(0), 3), " top eigenvalues:", np.round(eig[:k], 3))

m, Sigma = Z.mean(0), np.cov(Z, rowvar=False, bias=True)
T = 200
betas = np.linspace(1e-3, 0.05, T)
abars = np.cumprod(1 - betas)
print("abar_T =", round(float(abars[-1]), 5))


def eps_theta(zt, t):
    ab = abars[t]
    Sig_t = ab * Sigma + (1 - ab) * np.eye(k)
    return np.sqrt(1 - ab) * np.linalg.solve(Sig_t, (zt - np.sqrt(ab) * m).T).T


def sample(num, gen):
    z = gen.standard_normal((num, k))
    for t in range(T - 1, -1, -1):
        a = 1 - betas[t]
        z = (z - betas[t] / np.sqrt(1 - abars[t]) * eps_theta(z, t)) / np.sqrt(a)
        if t > 0:
            z = z + np.sqrt(betas[t]) * gen.standard_normal(z.shape)
    return z


zs = sample(4000, np.random.default_rng(1))
Xs = zs @ W + mean
cov_s = np.cov(Xs, rowvar=False, bias=True)
cov_x = np.cov(Xhat, rowvar=False, bias=True)
rel = np.linalg.norm(cov_s - cov_x) / np.linalg.norm(cov_x)
print("latent sample variances:", np.round(zs.var(0), 3))
print("relative covariance error in pixel space:", round(float(rel), 4))

# Cost: the denoiser is called T times on k numbers instead of D numbers.
print("values per step, pixel space:", D, " latent space:", k, " ratio:", D // k)
