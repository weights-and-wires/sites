"""Numbers for the "categorical projection" example (Section 2, Equation 3).

Run: python3 scripts/c51_projection.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def project(p, r, gamma_n, z):
    v_min, v_max = z[0], z[-1]
    dz = z[1] - z[0]
    tz = np.clip(r + gamma_n * z, v_min, v_max)
    b = (tz - v_min) / dz
    lo = np.floor(b).astype(int)
    hi = np.ceil(b).astype(int)
    m = np.zeros_like(p)
    for j in range(len(z)):
        if lo[j] == hi[j]:
            m[lo[j]] += p[j]
        else:
            m[lo[j]] += p[j] * (hi[j] - b[j])
            m[hi[j]] += p[j] * (b[j] - lo[j])
    return tz, b, m


if __name__ == "__main__":
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    r, gamma = 0.5, 0.9
    tz, b, m = project(p, r, gamma, z)
    print("support z          =", z)
    print("p                  =", p)
    print("shifted T z        =", tz, "(last atom clipped from 2.3)")
    print("fractional index b =", b)
    print("projected m        =", m)
    print("sum(m)             =", round(float(m.sum()), 6))
    print("mean before        =", round(float(z @ p), 4))
    print("mean of r + g*mean =", round(r + gamma * float(z @ p), 4))
    print("mean after         =", round(float(z @ m), 4))
    _, _, m_term = project(p, 1.5, 0.0, z)
    print("terminal (gamma_n=0, r=1.5) m =", m_term)
