"""Hinge losses (Equations 1 and 2) and spectral loss settings (Equations 4 and 5).

Run: python3 scripts/losses_example.py
"""

import numpy as np

# One discriminator (K = 1) with T = 4 logits per input.
D_real = np.array([1.5, 0.2, 2.0, -0.3])
D_fake = np.array([-0.5, 0.8, -2.0, 1.2])

real_term = np.maximum(0, 1 - D_real)
fake_term = np.maximum(0, 1 + D_fake)
print("max(0, 1 - D(x))     =", real_term.tolist(), "mean", float(real_term.mean()))
print("max(0, 1 + D(G(x)))  =", fake_term.tolist(), "mean", float(fake_term.mean()))
print("L_D (Equation 1)     =", float(real_term.mean() + fake_term.mean()))
gen_term = np.maximum(0, 1 - D_fake)
print("max(0, 1 - D(G(x)))  =", [round(float(g), 4) for g in gen_term], "mean", round(float(round(float(gen_term.mean()), 4)), 4))
print("L_adv_G (Equation 2) =", round(float(gen_term.mean()), 4))

print("\nThe hinge is flat once a logit passes the margin")
for d in (-1.0, 0.0, 0.5, 1.0, 2.0):
    print(f"D(G(x)) = {d:4.1f}: generator loss {max(0, 1 - d):.1f}, discriminator fake loss {max(0, 1 + d):.1f}")

print("\nAveraging over discriminators with different T_k (K = 2)")
logits = [np.array([0.0, 0.0]), np.array([1.0] * 6 + [-1.0] * 2)]
per_k = [float(np.maximum(0, 1 + l).mean()) for l in logits]
print("mean over t per discriminator:", per_k, "-> mean over k:", float(np.mean(per_k)))
flat = np.concatenate([np.maximum(0, 1 + l) for l in logits]).mean()
print("flat mean over all logits (not the paper's):", float(flat))

print("\nSpectral loss scales s = 2^6 ... 2^11 (Equations 4 and 5)")
print("s     hop   alpha_s = sqrt(s/2)")
for e in range(6, 12):
    s = 2 ** e
    print(f"{s:<5d} {s // 4:<5d} {np.sqrt(s / 2):.4f}")

