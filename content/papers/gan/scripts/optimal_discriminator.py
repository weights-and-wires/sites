"""Compute the numbers for the optimal-discriminator and JS-divergence example.

Run: python3 scripts/optimal_discriminator.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

p_data = np.array([0.4, 0.4, 0.2, 0.0])
p_g = np.array([0.1, 0.2, 0.3, 0.4])


def kl(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def jsd(p, q):
    m = (p + q) / 2
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def value(p, q, d):
    """V(G, D) for a discrete x, with the convention 0 log 0 = 0."""
    t1 = np.where(p > 0, p * np.log(np.where(p > 0, d, 1.0)), 0.0)
    t2 = np.where(q > 0, q * np.log(np.where(q > 0, 1 - d, 1.0)), 0.0)
    return float(np.sum(t1 + t2))


d_star = p_data / (p_data + p_g)
c = value(p_data, p_g, d_star)
m = (p_data + p_g) / 2
print("p_data              =", p_data.tolist())
print("p_g                 =", p_g.tolist())
print("D*(x)               =", [round(float(v), 4) for v in d_star])
print("C(G) = V(G, D*)     =", round(c, 4))
print("KL(p_data || m)     =", round(kl(p_data, m), 4))
print("KL(p_g || m)        =", round(kl(p_g, m), 4))
print("JSD(p_data || p_g)  =", round(jsd(p_data, p_g), 4))
print("-log 4              =", round(-np.log(4), 4))
print("-log 4 + 2 JSD      =", round(-np.log(4) + 2 * jsd(p_data, p_g), 4))

print()
print("A suboptimal D (constant 0.5)")
print("V(G, 0.5)           =", round(value(p_data, p_g, np.full(4, 0.5)), 4))

print()
print("Perfect generator, p_g = p_data")
sup = p_data > 0
print("D*(x) on the support =", (p_data[sup] / (p_data[sup] + p_data[sup])).tolist())
print("C(G)                =", round(value(p_data, p_data, np.full(4, 0.5)), 4))

print()
print("Disjoint supports")
a = np.array([0.5, 0.5, 0.0, 0.0])
b = np.array([0.0, 0.0, 0.5, 0.5])
d = np.array([1.0, 1.0, 0.0, 0.0])
print("JSD                 =", round(jsd(a, b), 4), "= log 2 =", round(float(np.log(2)), 4))
print("C(G)                =", round(value(a, b, d), 4))
