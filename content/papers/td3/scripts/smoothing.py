"""Numbers for the "Target policy smoothing" example (Section 5.3, Equations 13 and 14).

Run: python3 scripts/smoothing.py
"""

import numpy as np

# 1. The clipped noise and the clipped action, by hand.
sigma, c = 0.2, 0.5
low, high = -1.0, 1.0
mu_next = np.array([0.8, 0.8, 0.8])
z = np.array([0.5, 3.0, -4.0])          # standard normal draws
noise = np.clip(sigma * z, -c, c)
action = np.clip(mu_next + noise, low, high)
print("sigma * z:", sigma * z)
print("noise after clip(-c, c):", noise)
print("action after clip to bounds:", action)

# 2. Smoothing a narrow peak in the critic (Section 5.3: overfitting to narrow peaks).
def q_target(a):
    return 0.5 * a + 1.0 * np.exp(-(((a - 0.8) / 0.02) ** 2))


rng = np.random.default_rng(0)
eps = np.clip(0.2 * rng.standard_normal(100000), -0.5, 0.5)
print("Q'(0.8) without smoothing:", round(float(q_target(0.8)), 4))
print("mean Q'(0.8 + eps) with sigma 0.2, c 0.5:", round(float(q_target(np.clip(0.8 + eps, -1, 1)).mean()), 4))
print("Q' at the smooth part only, a = 0.8:", round(float(0.5 * 0.8), 4))
