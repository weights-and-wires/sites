import numpy as np


def gaussian_vf(x, x1, t, mu, dmu, sigma, dsigma):
    """u_t(x | x1) for the Gaussian path N(mu_t(x1), sigma_t^2 I) (Equation 15)."""
    return dsigma(t) / sigma(t) * (x - mu(t, x1)) + dmu(t, x1)


def vp_path(beta_min, beta_max):
    """Return (mu, dmu, sigma, dsigma) for the reversed VP path (Equation 18)."""

    def T(s):
        return beta_min * s + 0.5 * s**2 * (beta_max - beta_min)

    def beta(s):
        return beta_min + s * (beta_max - beta_min)

    def alpha(s):
        return np.exp(-0.5 * T(s))

    def mu(t, x1):
        return alpha(1.0 - t) * x1

    def sigma(t):
        return np.sqrt(1.0 - alpha(1.0 - t) ** 2)

    def dmu(t, x1):
        return 0.5 * beta(1.0 - t) * alpha(1.0 - t) * x1

    def dsigma(t):
        a = alpha(1.0 - t)
        return -a * (0.5 * beta(1.0 - t) * a) / sigma(t)

    return mu, dmu, sigma, dsigma
