"""Parametric loss and the efficient frontier (Section 3.3, Equations 2 and 4;
constants from Appendix D.2, Equation 10).

Run: python3 scripts/parametric_loss.py
"""
import numpy as np

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(n, d):
    return E + A / n**ALPHA + B / d**BETA


def frontier(c):
    g = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
    a, b = BETA / (ALPHA + BETA), ALPHA / (ALPHA + BETA)
    return g * (c / 6) ** a, (c / 6) ** b / g


a, b = BETA / (ALPHA + BETA), ALPHA / (ALPHA + BETA)
g = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
print(f"a = beta/(alpha+beta) = {a:.4f}, b = alpha/(alpha+beta) = {b:.4f}, G = {g:.4f}")

print("\nLoss terms at three (N, D) pairs")
for n, d in [(280e9, 300e9), (70e9, 1.4e12), (1e9, 20e9)]:
    print(f"N={n:.1e} D={d:.1e}  size term {A/n**ALPHA:.4f}  data term {B/d**BETA:.4f}  L = {loss(n, d):.4f}")

print("\nClosed-form frontier vs brute-force grid search")
for c in [1e19, 1e21, 5.76e23]:
    n, d = frontier(c)
    grid = np.logspace(np.log10(n) - 1, np.log10(n) + 1, 200001)
    best = grid[np.argmin(loss(grid, c / (6 * grid)))]
    print(f"C={c:.2e}  N_opt={n:.3e}  D_opt={d:.3e}  D/N={d/n:.2f}  L={loss(n, d):.4f}  grid N={best:.3e}")

print("\n10x more compute multiplies N_opt by", f"{10**a:.3f}", "and D_opt by", f"{10**b:.3f}")
