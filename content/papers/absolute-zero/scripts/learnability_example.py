"""Learnability reward (Section 3.1, Equation 4) for n = 8 solver rollouts."""

from azr_lib import learnability_reward

print("successes of 8 | mean solver reward | r_propose")
for k in [0, 1, 2, 4, 6, 7, 8]:
    rollouts = [1] * k + [0] * (8 - k)
    mean = k / 8
    print(f"{k:>14} | {mean:>18.3f} | {learnability_reward(rollouts):.3f}")
