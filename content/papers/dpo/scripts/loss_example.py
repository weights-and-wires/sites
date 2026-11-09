"""Compute the numbers for the DPO loss example (Equation 7).

Run: python3 content/papers/dpo/scripts/loss_example.py
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return np.logaddexp(0.0, -margin)  # -log sigmoid(margin), stable


# Sequence log-probabilities log pi(y | x) for one pair.
pi_w, pi_l = -12.0, -15.0
ref_w, ref_l = -13.0, -14.0
beta = 0.1

rhat_w = beta * (pi_w - ref_w)
rhat_l = beta * (pi_l - ref_l)
print("beta                         =", beta)
print("log ratio, winner            =", pi_w - ref_w)
print("log ratio, loser             =", pi_l - ref_l)
print("implicit reward, winner      =", round(rhat_w, 4))
print("implicit reward, loser       =", round(rhat_l, 4))
print("margin                       =", round(rhat_w - rhat_l, 4))
print("p(winner preferred)          =", round(float(sigmoid(rhat_w - rhat_l)), 4))
print("loss                         =", round(float(dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)), 4))

print()
print("Same log ratios, beta = 0.5")
print("margin                       =", round(0.5 * ((pi_w - ref_w) - (pi_l - ref_l)), 4))
print("loss                         =", round(float(dpo_loss(pi_w, pi_l, ref_w, ref_l, 0.5)), 4))

print()
print("Policy equals the reference (any log-probabilities)")
print("loss                         =", round(float(dpo_loss(-7.0, -9.0, -7.0, -9.0, beta)), 4), "= log 2 =", round(float(np.log(2)), 4))

print()
print("Wrong order: the policy moved toward the loser")
print("loss                         =", round(float(dpo_loss(-14.5, -13.5, ref_w, ref_l, beta)), 4))
print("margin                       =", round(beta * ((-14.5 - ref_w) - (-13.5 - ref_l)), 4))

print()
print("Numerical stability: margin = -1000")
with np.errstate(all="ignore"):
    naive = -np.log(sigmoid(np.float64(-1000.0)))
print("naive -log(sigmoid(m))       =", naive)
print("logaddexp(0, -m)             =", float(np.logaddexp(0.0, 1000.0)))

print()
print("Self-check: log ratios +0.8 (winner) and -0.5 (loser), beta = 0.1")
m = 0.1 * (0.8 - (-0.5))
print("margin                       =", round(m, 4))
print("loss                         =", round(float(np.logaddexp(0.0, -m)), 4))
print("p(winner preferred)          =", round(float(sigmoid(m)), 4))
