#!/usr/bin/env python3
"""The KL penalty and its adaptive coefficient (Section 4, Equation 8)."""

import numpy as np
from ppo_lib import kl_categorical, update_beta

p_old = np.array([0.5, 0.5])
p_new = np.array([0.6, 0.4])
print("KL[old, new] =", round(kl_categorical(p_old, p_new), 5))
print("KL[new, old] =", round(kl_categorical(p_new, p_old), 5))

d_targ = 0.01
print(f"\nd_targ = {d_targ}; thresholds: below {d_targ / 1.5:.5f} halve, above {d_targ * 1.5:.3f} double")
beta = 1.0
for d in [0.004, 0.012, 0.030, 0.0095, 0.0066]:
    new = update_beta(beta, d, d_targ)
    print(f"d = {d:.4f}  beta {beta:g} -> {new:g}")
    beta = new
