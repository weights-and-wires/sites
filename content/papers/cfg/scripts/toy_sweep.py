"""Sweep the guidance strength w on a 2-D mixture with exact scores.

Three classes sit on a circle of radius 2 with isotropic spread s = 1, as in
the paper's Figure 2 setup (Section 3.1). For z_lambda = alpha x + sigma eps
(Section 2, Equation 1) the exact noise predictions are closed forms, so no
network is needed. The script runs Algorithm 2 (Section 3.2) for class 0 with
the guided score of Equation 6 and prints how the samples change with w.

Run: python3 scripts/toy_sweep.py
"""

import numpy as np

MEANS = 2.0 * np.array([[np.cos(a), np.sin(a)] for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)])
S = 1.0
LAM_MIN, LAM_MAX = -20.0, 20.0


def alpha_sigma(lam):
    """alpha^2 = 1 / (1 + e^-lambda), sigma^2 = 1 - alpha^2 (Equation 1)."""
    a2 = 1.0 / (1.0 + np.exp(-lam))
    return np.sqrt(a2), np.sqrt(1.0 - a2)


def eps_cond(z, k, lam):
    """Exact eps*(z, c=k): -sigma times the score of N(alpha mu_k, (alpha^2 s^2 + sigma^2) I)."""
    a, sg = alpha_sigma(lam)
    return sg * (z - a * MEANS[k]) / (a**2 * S**2 + sg**2)


def eps_uncond(z, lam):
    """Exact eps*(z) for the equal-weight mixture of the three classes."""
    a, sg = alpha_sigma(lam)
    v = a**2 * S**2 + sg**2
    logits = np.stack([-np.sum((z - a * m) ** 2, axis=-1) / (2 * v) for m in MEANS], axis=-1)
    r = np.exp(logits - logits.max(axis=-1, keepdims=True))
    r /= r.sum(axis=-1, keepdims=True)
    mean = r @ (a * MEANS)
    return sg * (z - mean) / v


def lambdas(T, lam_min=LAM_MIN, lam_max=LAM_MAX):
    """lambda = -2 log tan(a u + b) at T uniformly spaced u (Section 2)."""
    b = np.arctan(np.exp(-lam_max / 2))
    a = np.arctan(np.exp(-lam_min / 2)) - b
    u = np.linspace(1.0, 0.0, T)  # lambda falls as u grows, so walk u downward
    return -2.0 * np.log(np.tan(a * u + b))


def sample(k, w, n, T, rng, v=0.0):
    """Algorithm 2 with the guided score (1 + w) eps(z, c) - w eps(z)."""
    lam = lambdas(T)
    z = rng.standard_normal((n, 2))
    for t in range(T):
        a, sg = alpha_sigma(lam[t])
        eps = (1 + w) * eps_cond(z, k, lam[t]) - w * eps_uncond(z, lam[t])
        x = (z - sg * eps) / a
        if t == T - 1:
            return x
        a2, sg2 = alpha_sigma(lam[t + 1])
        r = np.exp(lam[t] - lam[t + 1])
        mean = r * (a2 / a) * z + (1 - r) * a2 * x
        var_tilde = (1 - r) * sg2**2
        var_fwd = (1 - r) * sg**2
        var = var_tilde ** (1 - v) * var_fwd**v
        z = mean + np.sqrt(var) * rng.standard_normal(z.shape)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, T = 20000, 256
    mu0 = MEANS[0]
    away = mu0 / np.linalg.norm(mu0)  # direction away from the other two classes
    print("w     mean_along  std_along  std_across  frac_closest_to_class0")
    for w in (0.0, 0.5, 1.0, 2.0, 4.0):
        x = sample(0, w, n, T, rng)
        along = x @ away
        across = x @ np.array([-away[1], away[0]])
        d = np.linalg.norm(x[:, None, :] - MEANS[None], axis=-1)
        frac = float(np.mean(d.argmin(axis=1) == 0))
        print(f"{w:<5} {along.mean():10.3f} {along.std():10.3f} {across.std():11.3f} {frac:22.4f}")
