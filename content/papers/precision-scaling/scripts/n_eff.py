"""Effective parameter count N_eff (Section 4.1 and Section 4.2, Equation 4),
with the constants of Appendix K, Table 2.

Run: python3 scripts/n_eff.py
"""
import numpy as np

GAMMA = {"w": 2.6745, "a": 2.2102, "kv": 0.9578}  # Table 2: gamma_w, gamma_i, gamma_kv
OFFSET = {"w": 0.3037, "a": 1.4072, "kv": 2.4185}  # Table 2: n_w, n_i, n_kv


def factor(p, gamma, offset=0.0):
    """One factor 1 - exp(-p / gamma + offset) of Equation 4."""
    return 1.0 - np.exp(-p / gamma + offset)


print("Weights only, Equation 4 without offsets (gamma_w = 2.6745)")
for p in [2, 3, 4, 6, 8, 16]:
    print(f"P_w={p:>2}  1 - exp(-P/gamma_w) = {factor(p, GAMMA['w']):.4f}")

gbar = sum(GAMMA.values()) / 3
print(f"\ngamma_bar = mean of the three gammas = {gbar:.4f}")
print("Uniform precision, N_eff / N = (1 - exp(-P/gamma_bar))^3")
for p in [3, 4, 6, 8, 16]:
    print(f"P={p:>2}  N_eff/N = {factor(p, gbar) ** 3:.4f}")

print("\nFull Equation 4 with the Table 2 offsets, all three precisions equal to P")
for p in [4, 5, 6, 8, 16]:
    fs = {k: factor(p, GAMMA[k], OFFSET[k]) for k in GAMMA}
    prod = np.prod(list(fs.values()))
    print(f"P={p:>2}  w {fs['w']:.4f}  a {fs['a']:.4f}  kv {fs['kv']:.4f}  N_eff/N = {prod:.4f}")

print("\nOrder matters: N_eff / N for N = 1e9 with the three gammas of Table 2, no offsets")
for pw, pa, pkv in [(8, 4, 8), (4, 8, 8), (8, 8, 4)]:
    prod = factor(pw, GAMMA["w"]) * factor(pa, GAMMA["a"]) * factor(pkv, GAMMA["kv"])
    print(f"(P_w, P_a, P_kv) = ({pw}, {pa}, {pkv})  N_eff/N = {prod:.4f}")

print("\nActivation factor at P_a = 3 with the offset:", f"{factor(3, GAMMA['a'], OFFSET['a']):.4f}", "(negative: the fit breaks down)")
