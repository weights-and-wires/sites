"""Numbers for the "multi-step returns" example (Section 2, Equation 2).

Run: python3 scripts/nstep_example.py
"""

import numpy as np

gamma = 0.9
rewards = [1.0, 0.0, 2.0]  # R_{t+1}, R_{t+2}, R_{t+3}
q_boot = 5.0  # max_a' q(S_{t+3}, a')

n = len(rewards)
ret = sum(gamma**k * r for k, r in enumerate(rewards))
print(f"R_t^(3)                     = {ret:.4f}")
print(f"gamma^3                     = {gamma**n:.4f}")
print(f"bootstrap term gamma^3 * 5  = {gamma**n * q_boot:.4f}")
print(f"3-step target               = {ret + gamma**n * q_boot:.4f}")
one_step = rewards[0] + gamma * q_boot
print(f"1-step target (same q)      = {one_step:.4f}")

# The episode ends on the second reward: the window holds two rewards and no bootstrap.
short = [1.0, 2.0]
ret2 = sum(gamma**k * r for k, r in enumerate(short))
print(f"terminal after 2 rewards    = {ret2:.4f}  (no bootstrap term)")
