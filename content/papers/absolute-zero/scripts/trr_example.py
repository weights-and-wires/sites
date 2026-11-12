"""Task-relative normalization (Section 3.3.5, Equation 8) on a toy batch."""

import numpy as np

from azr_lib import task_relative_advantages

rewards = [1, 0, 0, 1, 0.75, 0.5, 0.125, 0.0, 1, 1, 0, 1]
tasks = ["ded"] * 4 + ["ded"] * 4 + ["abd"] * 4
roles = ["solve"] * 4 + ["propose"] * 4 + ["solve"] * 4
adv = task_relative_advantages(rewards, tasks, roles)
for key in [("ded", "solve"), ("ded", "propose"), ("abd", "solve")]:
    idx = [i for i, k in enumerate(zip(tasks, roles)) if k == key]
    g = np.array(rewards)[idx]
    print(key, "rewards", g, "mean", round(float(g.mean()), 4), "std", round(float(g.std()), 4))
    print("   advantages", np.round(adv[idx], 4))
flat = (np.array(rewards) - np.mean(rewards)) / np.std(rewards)
print("one shared baseline instead:", np.round(flat[:4], 4), "for the first group")
