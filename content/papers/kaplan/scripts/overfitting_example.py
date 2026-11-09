"""Evaluate L(N, D) from Equation 1.5 with the fit of Table 2, and the data rule of Equation 4.4.

Run: python3 scripts/overfitting_example.py
"""

ALPHA_N, ALPHA_D, N_C, D_C = 0.076, 0.103, 6.4e13, 1.8e13  # Table 2


def loss(n, d):
    return ((N_C / n) ** (ALPHA_N / ALPHA_D) + D_C / d) ** ALPHA_D


def loss_n(n):  # D -> infinity, Equation 3.1 with Table 2 constants
    return (N_C / n) ** ALPHA_N


def loss_d(d):  # N -> infinity
    return (D_C / d) ** ALPHA_D


print(f"alpha_N / alpha_D = {ALPHA_N / ALPHA_D:.4f}")
print()
n = 1e9
print(f"N = {n:.0e}")
print(f"  L(N) with infinite data = {loss_n(n):.4f}")
for d in [1e8, 1e9, 1e10, 2.2e10, 1e12]:
    l = loss(n, d)
    print(f"  D = {d:.1e}: L(N, D) = {l:.4f}, deltaL = L/L(N, inf) - 1 = {l / loss_n(n) - 1:.4f}")

print()
print("Limits at N = 1e9, D = 1e10:")
print(f"  L(1e9, 1e30)  = {loss(1e9, 1e30):.4f}  vs L(N)        = {loss_n(1e9):.4f}")
print(f"  L(1e30, 1e10) = {loss(1e30, 1e10):.4f}  vs L(D)        = {loss_d(1e10):.4f}")

print()
print("Equation 4.4: D >= 5e3 * N**0.74")
for n in [1e8, 1e9, 8e9]:
    print(f"  N = {n:.0e}: D >= {5e3 * n**0.74:.3e} tokens")
print(f"  growth in D for an 8x larger model: {8**0.74:.3f}x")
print(f"  alpha_N / alpha_D from Section 1.2 (0.076 / 0.095)      = {0.076 / 0.095:.3f}")
