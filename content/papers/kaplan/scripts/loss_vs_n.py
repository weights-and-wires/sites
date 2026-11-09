"""Evaluate L(N) = (N_c / N)**alpha_N (Equation 1.1) for a few model sizes.

Run: python3 scripts/loss_vs_n.py
"""

ALPHA_N, N_C = 0.076, 8.8e13  # Section 1.2, Equation 1.1

print(f"{'N':>10} {'L(N)':>8}")
for n in [1e6, 1e7, 1e8, 1e9, 1e10]:
    print(f"{n:10.0e} {(N_C / n) ** ALPHA_N:8.4f}")

print()
print(f"2x parameters multiply the loss by 2**-alpha_N  = {2 ** -ALPHA_N:.4f}")
print(f"10x parameters multiply the loss by 10**-alpha_N = {10 ** -ALPHA_N:.4f}")
print(f"Parameters needed to cut the loss by 10% (factor 0.9): {0.9 ** (-1 / ALPHA_N):.2f}x")
print(f"Parameters needed to halve the loss: {0.5 ** (-1 / ALPHA_N):.3e}x")
