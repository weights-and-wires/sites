import numpy as np


def project_distribution(p, r, gamma_n, z):
    """Project the distribution (r + gamma_n * z, p) back onto the fixed atoms z."""
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
