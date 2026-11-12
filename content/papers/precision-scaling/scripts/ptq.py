"""Post-training quantization degradation and the critical dataset size
(Section 3.1, Equation 2; Appendix E.1, Equations 12 and 13). Constants from
Appendix K, Table 2. The unlabeled "gamma = 0.5907" row is read here as
gamma_post (inferred from its position after gamma_D and gamma_N).

Run: python3 scripts/ptq.py
"""
import numpy as np

B, BETA = 1.806e4, 0.4965
C_T, GAMMA_D, GAMMA_N, GAMMA_POST = 0.0598, 0.5068, 0.3439, 0.5907


def delta(n, d, p_post):
    return C_T * d**GAMMA_D / n**GAMMA_N * np.exp(-p_post / GAMMA_POST)


def d_crit(n, p_post):
    return (BETA * B * n**GAMMA_N * np.exp(p_post / GAMMA_POST) / (GAMMA_D * C_T)) ** (1 / (GAMMA_D + BETA))


print("delta_PTQ at P_post = 4 (nats)")
for n, d in [(30e6, 1.5e9), (30e6, 26e9), (220e6, 26e9)]:
    print(f"N={n:.1e} D={d:.1e} D/N={d / n:>6.0f}  delta={delta(n, d, 4):.4f}")

print("\nTenfold more tokens multiplies delta by", f"{10**GAMMA_D:.3f}", "and tenfold more parameters by", f"{10**-GAMMA_N:.3f}")
print("One more bit multiplies delta by", f"{np.exp(-1 / GAMMA_POST):.4f}")

print("\nCritical dataset size D_crit")
for n in [30e6, 220e6, 8e9]:
    for p in [4, 5, 8]:
        dc = d_crit(n, p)
        print(f"N={n:.1e} P_post={p}  D_crit={dc:.3e}  D_crit/N={dc / n:.3e}")

n, p = 30e6, 5
dc = d_crit(n, p)
eps = 1e-4
f = lambda d: B * d**-BETA + delta(n, d, p)
slope = (f(dc * (1 + eps)) - f(dc * (1 - eps))) / (2 * eps * dc)
print(f"\nCheck at N=30M, P_post=5: slope of B D^-beta + delta at D_crit = {slope:.2e}")
