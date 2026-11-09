"""Compute-optimal allocation from Table 5 and Equation 1.3.

Run: python3 scripts/compute_optimal.py
"""

# Table 6 (Appendix A): compute-efficient values as power laws in C_min (PF-days).
N_E, P_N = 1.3e9, 0.73
D_E, P_D = 2e10, 0.27
S_E, P_S = 5.4e3, 0.03
B_E, P_B = 2.0e6, 0.24
# Equation 1.3 and Table 4
ALPHA_CMIN, CMIN_C = 0.050, 3.1e8


def loss(c):
    return (CMIN_C / c) ** ALPHA_CMIN


print(f"{'C_min (PF-days)':>16} {'N_opt':>10} {'D_opt':>10} {'S_min':>8} {'B (tokens)':>11} {'L(C_min)':>9}")
for c in [1e-3, 1e-2, 1e-1, 1, 10, 1e2]:
    print(f"{c:16.0e} {N_E * c**P_N:10.2e} {D_E * c**P_D:10.2e} {S_E * c**P_S:8.0f} {B_E * c**P_B:11.2e} {loss(c):9.4f}")

print()
print(f"10x compute multiplies N_opt by {10**P_N:.3f}, D_opt by {10**P_D:.3f}")
print(f"10x compute multiplies the loss by {10**-ALPHA_CMIN:.4f}")
print(f"1e9x compute multiplies N_opt by {1e9**P_N:.3e}, B by {1e9**P_B:.1f}, S by {1e9**P_S:.2f}")
print(f"Exponents sum: pN + pB + pS = {P_N + P_B + P_S:.2f}  (C = 6 N B S)")

a_s, a_b, a_n = 0.76, 0.21, 0.076
print()
print(f"Equation 1.8 with alpha_S=0.76, alpha_B=0.21, alpha_N=0.076: alpha_C = {1 / (1 / a_s + 1 / a_b + 1 / a_n):.4f}")
print(f"  implied N exponent alpha_C / alpha_N = {1 / (1 / a_s + 1 / a_b + 1 / a_n) / a_n:.3f}")

# Data growth: Equation 6.6 versus Equation 6.7 exponents.
print()
print("Data needed to avoid overfitting grows as C**0.54 (Eq. 6.6); one epoch grows as C**0.26 (Eq. 6.7)")
print(f"  D(C) = 4e10 * C**0.26: at C = 1 PF-day {4e10:.1e} tokens, at 1e4 PF-days {4e10 * 1e4**0.26:.3e}")
print(f"  N**0.74 at C = 1e4: {(N_E * 1e4**P_N) ** 0.74 * 5e3:.3e} tokens needed by Eq. 4.4")

# Equations 5.2 to 5.5: correcting a run at a fixed batch size for B_crit(L).
B_STAR, ALPHA_B = 2.1e8, 0.21  # Equation 1.4 with the constant of Table 5
L_TARGET, B, S = 3.0, 2**19, 1e5
b_crit = B_STAR / L_TARGET ** (1 / ALPHA_B)
print()
print(f"Target loss {L_TARGET}: B_crit = {b_crit:.4e} tokens, run batch B = 2**19 = {B}")
print(f"  S_min = S / (1 + B_crit / B) = {S / (1 + b_crit / B):.0f} steps for S = {S:.0e}")
print(f"  C_min = C / (1 + B / B_crit) = {1 / (1 + B / b_crit):.4f} * C")
print(f"  at B = B_crit: steps = 2 * S_min, compute = 2 * C_min")
