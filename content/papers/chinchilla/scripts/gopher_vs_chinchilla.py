"""Gopher versus Chinchilla at one compute budget (Section 4, Table 1, Table 3,
Table 4, Appendix F). Constants from Appendix D.2, Equation 10.

Run: python3 scripts/gopher_vs_chinchilla.py
"""
E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(n, d):
    return E + A / n**ALPHA + B / d**BETA


for name, n, d in [("Gopher", 280e9, 300e9), ("Chinchilla", 70e9, 1.4e12)]:
    print(f"{name:11s} N={n:.2e} D={d:.2e}  6ND={6*n*d:.3e}  D/N={d/n:5.1f}  L_hat={loss(n, d):.4f}")

print()
print("Same 6ND budget? ratio Chinchilla/Gopher =", f"{6*70e9*1.4e12/(6*280e9*300e9):.3f}")
print("Inference FLOPs per token (2N) ratio Gopher/Chinchilla =", f"{280/70:.1f}")

# Table 3 rows (Approach 1): tokens per parameter.
print("\nTable 3 rows: tokens per parameter and 6ND against the listed FLOPs")
for n, c, d in [(1e9, 1.21e20, 20.2e9), (10e9, 1.23e22, 205.1e9), (67e9, 5.76e23, 1.5e12), (175e9, 3.85e24, 3.7e12), (1e12, 1.27e26, 21.2e12)]:
    print(f"N={n:.0e} tokens={d:.2e}  D/N={d/n:5.1f}  6ND={6*n*d:.2e}  listed C={c:.2e}")

# Kaplan versus Hoffmann scaling for a 10x compute increase (Section 1).
print(f"\n10x compute: Kaplan N x{10**0.73:.2f}, D x{10**0.27:.2f}; Hoffmann Approach 1 N x{10**0.5:.2f}, D x{10**0.5:.2f}")

# Table 2 exponents: a + b = 1 for each approach.
for name, a, b in [("Approach 1", .50, .50), ("Approach 2", .49, .51), ("Approach 3", .46, .54), ("Kaplan", .73, .27)]:
    print(f"{name:11s} a={a:.2f} b={b:.2f} a+b={a+b:.2f}  tokens per parameter grow x{10**(b-a):.2f} per 10x compute")
