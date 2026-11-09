import numpy as np

NULL = -1
MU = np.array([-2.0, 2.0])  # class means of the 1-D toy data, x | c ~ N(mu_c, 1)


def alpha_sigma(lam):
    """Finished for you. Equation 1."""
    alpha_sq = 1.0 / (1.0 + np.exp(-lam))
    return np.sqrt(alpha_sq), np.sqrt(1.0 - alpha_sq)


def lambdas(T, lam_min=-20.0, lam_max=20.0):
    """Finished for you. T increasing log-SNR values, lam_1 = lam_min and lam_T = lam_max."""
    b = np.arctan(np.exp(-lam_max / 2))
    a = np.arctan(np.exp(-lam_min / 2)) - b
    u = np.linspace(1.0, 0.0, T)
    return -2.0 * np.log(np.tan(a * u + b))


def toy_eps_model(z, c, lam):
    """Finished for you. The exact eps*(z, c) for the toy data; c == NULL gives eps*(z).

    z has shape (n,), c is an int class label or NULL, lam is a float.
    """
    alpha, sigma = alpha_sigma(lam)
    v = alpha**2 + sigma**2
    if c != NULL:
        return sigma * (z - alpha * MU[c]) / v
    logit = -(z - alpha * MU[1]) ** 2 / (2 * v) + (z - alpha * MU[0]) ** 2 / (2 * v)
    r1 = 1.0 / (1.0 + np.exp(-logit))
    mean = alpha * (r1 * MU[1] + (1 - r1) * MU[0])
    return sigma * (z - mean) / v



def sample_cfg(eps_model, c, w, n, T, rng, v=0.0):
    """Algorithm 2: classifier-free guided ancestral sampling of n points for class c.

    eps_model(z, c, lam) returns the noise prediction; c == NULL means unconditional.
    Draw z_1 ~ N(0, I), then for t = 1..T form the guided score (Equation 6), compute
    x_t = (z_t - sigma * eps) / alpha, and move to lambda_{t+1} with Equations 3 and 4.
    At t = T return x_T and add no noise. The variance is
    (sigma_tilde^2)^(1 - v) * (sigma_{lam|lam'}^2)^v, with
    sigma_tilde^2 = (1 - e^(lam - lam')) sigma_{lam'}^2 and
    sigma_{lam|lam'}^2 = (1 - e^(lam - lam')) sigma_lam^2.
    Call rng.standard_normal exactly T times in all: once for z_1 and T - 1 times for the steps.
    """
    lam = lambdas(T)
    z = rng.standard_normal(n)
    for t in range(T):
        alpha, sigma = alpha_sigma(lam[t])
        eps = (1.0 + w) * eps_model(z, c, lam[t]) - w * eps_model(z, NULL, lam[t])
        x = (z - sigma * eps) / alpha
        if t == T - 1:
            return x
        alpha2, sigma2 = alpha_sigma(lam[t + 1])
        r = np.exp(lam[t] - lam[t + 1])
        mean = r * (alpha2 / alpha) * z + (1.0 - r) * alpha2 * x
        var_tilde = (1.0 - r) * sigma2**2
        var_fwd = (1.0 - r) * sigma**2
        var = var_tilde ** (1.0 - v) * var_fwd**v
        z = mean + np.sqrt(var) * rng.standard_normal(n)
