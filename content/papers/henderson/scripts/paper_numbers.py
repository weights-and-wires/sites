"""Arithmetic on numbers printed in the paper's Table 1.

The table entries are copied from the paper (mean +/- standard error over 5
trials, after 2M timesteps). This script only computes the differences and
ratios that the page quotes.

Run: python3 paper_numbers.py
"""

# PPO, Hopper-v1, policy (64, 64) and (400, 300).
ppo_small, ppo_small_se = 2790, 62
ppo_big, ppo_big_se = 61, 33
diff = ppo_small - ppo_big
print(f"PPO Hopper: difference {diff}, which is {diff / ppo_small_se:.1f} times the larger standard error ({ppo_small_se})")

# TRPO, HalfCheetah-v1, policy (400, 300), three codebases.
cells = {"original code": (1791, 224), "rllab": (738, 240), "Baselines (PPO paper)": (-155, 188)}
means = [m for m, _ in cells.values()]
ses = [s for _, s in cells.values()]
span = max(means) - min(means)
print(f"TRPO HalfCheetah: means span {span}, which is {span / max(ses):.1f} times the largest standard error ({max(ses)})")
