"""IsoFLOP profile for the Approach 3 constants (Section 3.2, Figure 3, and
Section 3.3, Figure 4 right), with the Approach 2 parabola fit in log N.

Run: python3 scripts/isoflop.py
"""
import numpy as np

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(n, d):
    return E + A / n**ALPHA + B / d**BETA


C = 1e21
sizes = np.array([0.4e9, 0.8e9, 1.6e9, 3.2e9, 6.4e9, 12.8e9])
print(f"IsoFLOP slice at C = {C:.0e}")
for n in sizes:
    d = C / (6 * n)
    print(f"N={n:.2e}  D={d:.2e}  D/N={d/n:6.1f}  L={loss(n, d):.4f}")

# Approach 2: fit a parabola to L against log N, read the minimum.
x = np.log(sizes)
y = loss(sizes, C / (6 * sizes))
c2, c1, c0 = np.polyfit(x, y, 2)
n_par = np.exp(-c1 / (2 * c2))
g = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
n_star = g * (C / 6) ** (BETA / (ALPHA + BETA))
print(f"\nparabola minimum N = {n_par:.3e}; closed form N_opt = {n_star:.3e}; ratio {n_par/n_star:.3f}")
print(f"parabola minimum loss = {np.polyval([c2, c1, c0], np.log(n_par)):.4f}; true minimum = {loss(n_star, C/(6*n_star)):.4f}")
