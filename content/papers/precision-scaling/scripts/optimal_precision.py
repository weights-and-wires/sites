"""Compute-optimal training precision (Section 4.3.2, Equations 5 to 7).

With alpha = beta and the cost C = (6/16) N D P, minimizing L over N at fixed P
leaves P / f(P)^3 to minimize, where f(P) = 1 - exp(-P/gamma_bar). This page
derives the stationarity condition exp(x) - 1 = 3x for x = P / gamma_bar. It
does not depend on C, A, B, E, or alpha.

Run: python3 scripts/optimal_precision.py
"""
import numpy as np

A, B, E, ALPHA = 4.299e3, 1.806e4, 2.7648, 0.4965
GAMMA = {"w": 2.6745, "a": 2.2102, "kv": 0.9578}
OFFSET = {"w": 0.3037, "a": 1.4072, "kv": 2.4185}
GBAR = sum(GAMMA.values()) / 3

x = 2.0
for _ in range(50):
    x -= (np.exp(x) - 1 - 3 * x) / (np.exp(x) - 3)
print(f"root of exp(x) - 1 = 3x: x* = {x:.4f}")
print(f"gamma_bar = {GBAR:.4f}  ->  P* = x* gamma_bar = {x * GBAR:.3f} bits (Equation 4 without offsets)")


def fraction(p, offsets):
    """Product of the three factors, or -inf where a factor is not positive."""
    f = np.ones_like(p)
    for k in GAMMA:
        t = 1.0 - np.exp(-p / GAMMA[k] + (OFFSET[k] if offsets else 0.0))
        f = np.where((t > 0) & (f > 0), f * t, -np.inf)
    return f


p_grid = np.linspace(0.5, 16, 31001)
for offsets, label in [(False, "no offsets (three gammas)"), (True, "Table 2 offsets")]:
    f = fraction(p_grid, offsets)
    obj = np.where(f > 0, p_grid / np.where(f > 0, f, 1.0), np.inf)
    print(f"{label}: argmin of P / (f_w f_a f_kv) on a grid = {p_grid[obj.argmin()]:.3f} bits")


def best_over_n(c, p, offsets=False):
    n = np.logspace(6, 13, 20000)
    d = c / (6 / 16 * n * p)
    f = fraction(np.array([float(p)]), offsets)[0]
    if f <= 0:
        return np.inf, np.nan
    loss = A * (n * f) ** -ALPHA + B * d**-ALPHA + E
    return loss.min(), n[loss.argmin()]


for offsets, label in [(False, "no offsets"), (True, "Table 2 offsets")]:
    print(f"\nBrute force over N and P in steps of 0.5 bit, three precisions equal, {label}")
    for c in [1e19, 1e21, 1e23]:
        res = [(best_over_n(c, p, offsets)[0], p) for p in np.arange(2, 16.01, 0.5)]
        loss, p = min(res)
        n_opt = best_over_n(c, p, offsets)[1]
        print(f"C={c:.0e}  best P={p:.1f}  N={n_opt:.3e}  L={loss:.4f}")

print("\nOptimal N at C = 1e21 for several P (Table 2 offsets)")
for p in [5, 7, 8, 16]:
    loss, n_opt = best_over_n(1e21, p, True)
    print(f"P={p:>2}  N*={n_opt:.3e}  D*={1e21 / (6 / 16 * n_opt * p):.3e}  L={loss:.4f}")
