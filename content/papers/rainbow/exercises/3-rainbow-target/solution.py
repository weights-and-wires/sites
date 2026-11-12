import numpy as np


def project_distribution(p, r, gamma_n, z):
    """Project the distribution (r + gamma_n * z, p) back onto the fixed atoms z (given)."""
    p = np.asarray(p, dtype=float)
    z = np.asarray(z, dtype=float)
    dz = z[1] - z[0]
    tz = np.clip(r + gamma_n * z, z[0], z[-1])
    b = (tz - z[0]) / dz
    lo = np.floor(b + 1e-9).astype(int)
    hi = np.ceil(b - 1e-9).astype(int)
    m = np.zeros_like(p)
    for j in range(len(z)):
        if lo[j] == hi[j]:
            m[lo[j]] += p[j]
        else:
            m[lo[j]] += p[j] * (hi[j] - b[j])
            m[hi[j]] += p[j] * (b[j] - lo[j])
    return m


def rainbow_target(p_online, p_target, z, rewards, done, gamma):
    """Return (m, a_star): the projected m-step target distribution and the chosen action."""
    z = np.asarray(z, dtype=float)
    a_star = int(np.argmax(np.asarray(p_online) @ z))
    ret = sum(gamma**k * r for k, r in enumerate(rewards))
    gamma_n = 0.0 if done else gamma ** len(rewards)
    m = project_distribution(np.asarray(p_target)[a_star], ret, gamma_n, z)
    return m, a_star


def kl_priority(m, p_pred, omega):
    """Return KL(m || p_pred) ** omega, with 0 log 0 = 0."""
    m = np.asarray(m, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    mask = m > 0
    kl = float(np.sum(m[mask] * np.log(m[mask] / p_pred[mask])))
    return max(kl, 0.0) ** omega
