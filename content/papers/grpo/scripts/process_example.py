"""Process-supervision advantages (Section 4.1.3) on a group of G = 3 outputs."""

import numpy as np

from grpo_lib import process_advantages

steps = [np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])]
flat = np.concatenate(steps)
print("step rewards:", [s.tolist() for s in steps])
print("mean, std over all", len(flat), "steps:", round(float(flat.mean()), 4), round(float(flat.std()), 4))
norm = [(s - flat.mean()) / flat.std() for s in steps]
for i, n in enumerate(norm):
    print(f"normalized rewards, output {i + 1}:", np.round(n, 4))
for i, a in enumerate(process_advantages(steps)):
    print(f"advantages by step, output {i + 1}:", np.round(a, 4))
