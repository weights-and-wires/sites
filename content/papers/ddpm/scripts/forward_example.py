"""Compute the numbers for the forward-process examples.

Run: python3 scripts/forward_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Part 1: a four-step schedule small enough to check by hand.
betas = np.array([0.1, 0.2, 0.3, 0.4])
alphas = 1 - betas
alpha_bars = np.cumprod(alphas)
print("betas       =", betas.tolist())
print("alphas      =", alphas.tolist())
print("alpha_bars  =", alpha_bars.tolist())
print("sqrt(ab_3)  =", float(np.sqrt(alpha_bars[2])))
print("1 - ab_3    =", float(1 - alpha_bars[2]))

# Stepwise noising (Equation 2) versus the closed form (Equation 4), x0 = 2.
rng = np.random.default_rng(0)
n = 400_000
x0 = 2.0
x = np.full(n, x0)
for t in range(3):
    x = np.sqrt(alphas[t]) * x + np.sqrt(betas[t]) * rng.standard_normal(n)
print("stepwise t=3: mean %.4f var %.4f" % (x.mean(), x.var()))
print("closed form : mean %.4f var %.4f" % (np.sqrt(alpha_bars[2]) * x0, 1 - alpha_bars[2]))

# Part 2: the paper's schedule (Section 4): T = 1000, beta from 1e-4 to 0.02.
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alpha_bars = np.cumprod(1 - betas)
for t in (1, 100, 500, 1000):
    ab = alpha_bars[t - 1]
    print(f"t={t:4d}  alpha_bar={ab:.6f}  signal sqrt(ab)={np.sqrt(ab):.4f}  noise sqrt(1-ab)={np.sqrt(1 - ab):.4f}")

# L_T = KL(q(x_T|x0) || N(0, I)) per dimension, in bits, for a coordinate x0 = 1
# (data scaled to [-1, 1] sit at most at distance 1 from 0).
ab = alpha_bars[-1]
for x0 in (1.0,):
    mu2 = ab * x0**2
    var = 1 - ab
    kl_nats = 0.5 * (mu2 + var - 1 - np.log(var))
    print(f"L_T for x0={x0}: {kl_nats:.3e} nats = {kl_nats / np.log(2):.3e} bits per dimension")
