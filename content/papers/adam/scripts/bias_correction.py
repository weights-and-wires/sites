"""Show the initialization bias of the moving averages and its correction
(Section 3, Equations 1 to 4; Section 6.4).

Run: python3 scripts/bias_correction.py
"""

import numpy as np

alpha, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

print("Constant gradient g = 1 for 10 steps; step size divided by alpha")
print("t   corrected   uncorrected")
m = v = 0.0
for t in range(1, 11):
    m = beta1 * m + (1 - beta1)
    v = beta2 * v + (1 - beta2)
    corrected = (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
    uncorrected = m / (np.sqrt(v) + eps)
    print(f"{t:<3} {corrected:9.4f}   {uncorrected:9.4f}")

print()
print("Decay factors 1 - beta2^t that Equation 4 divides out")
for t in (1, 10, 100, 1000, 5000):
    print(f"t={t:<5} 1-beta2^t = {1 - beta2**t:.6f}")

print()
print("Check Equation 4 with seeded g ~ N(0, 1), so E[g^2] = 1")
rng = np.random.default_rng(0)
trials = 20000
g = rng.normal(size=(trials, 50))
v = np.zeros(trials)
for t in range(1, 51):
    v = beta2 * v + (1 - beta2) * g[:, t - 1] ** 2
    if t in (1, 10, 50):
        print(
            f"t={t:<3} mean v_t = {v.mean():.5f}   1-beta2^t = {1 - beta2**t:.5f}   "
            f"mean v_t / (1-beta2^t) = {v.mean() / (1 - beta2**t):.4f}"
        )
