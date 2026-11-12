"""Chinchilla loss with N replaced by N_eff (Section 4.1, Equation 3; Section 4.2,
Equation 4; Section 5, Equation 11 without the PTQ term). Constants from
Appendix K, Table 2, with alpha = beta (Appendix K).

Run: python3 scripts/loss_precision.py
"""
import numpy as np

A, B, E, ALPHA = 4.299e3, 1.806e4, 2.7648, 0.4965
GAMMA = {"w": 2.6745, "a": 2.2102, "kv": 0.9578}
OFFSET = {"w": 0.3037, "a": 1.4072, "kv": 2.4185}


def n_eff(n, p):
    f = 1.0
    for k in GAMMA:
        f *= 1.0 - np.exp(-p / GAMMA[k] + OFFSET[k])
    return n * f


def loss(n, d, p):
    return A * n_eff(n, p) ** -ALPHA + B * d**-ALPHA + E


print("Loss of an N = 220M model on D = 26B tokens, all precisions equal to P")
n, d = 220e6, 26e9
for p in [5, 6, 8, 16]:
    print(f"P={p:>2}  N_eff={n_eff(n, p):.3e}  size term={A * n_eff(n, p) ** -ALPHA:.4f}  data term={B * d ** -ALPHA:.4f}  L={loss(n, d, p):.4f}")

print("\nHow much bigger must a 5-bit model be to match the loss of an 8-bit model?")
ratio = n_eff(1.0, 8) / n_eff(1.0, 5)
print(f"N_eff(8) / N_eff(5) at equal N = {ratio:.3f}")
n8 = 100e6
n5 = n8 * ratio
print(f"8-bit N={n8:.3e}  L={loss(n8, d, 8):.4f}   5-bit N={n5:.3e}  L={loss(n5, d, 5):.4f}")
