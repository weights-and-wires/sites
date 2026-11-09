"""Show what the KL coefficient does on a three-response toy problem.

For the objective E[r(y)] - beta * KL(pi || pi_SFT) over a discrete set of
responses, the maximizer is pi*(y) proportional to pi_SFT(y) * exp(r(y) / beta).
This is a standard closed form. The InstructGPT paper doesn't derive it.

Run: python3 content/papers/instructgpt/scripts/kl_optimum.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

pi_sft = np.array([0.70, 0.25, 0.05])
reward = np.array([0.0, 1.0, 2.0])


def optimum(beta):
    w = pi_sft * np.exp(reward / beta)
    return w / w.sum()


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


print("pi_SFT =", pi_sft.tolist(), " rewards =", reward.tolist())
print("beta    pi*                          E[r]    KL(pi*||pi_SFT)")
for beta in (10.0, 1.0, 0.5, 0.1, 0.02):
    p = optimum(beta)
    print(f"{beta:<6}  {np.array2string(p, precision=4, suppress_small=True)}  {float(p @ reward):.4f}  {kl(p, pi_sft):.4f}")
print(f"E[r] under pi_SFT = {float(pi_sft @ reward):.4f}")
