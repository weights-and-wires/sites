#!/usr/bin/env python3
"""Numbers for the ensemble disagreement rule (Section 2.2.4)."""

import numpy as np
from pref_lib import normalize_rewards, select_queries, sigmoid

ws = [np.array([1.0]), np.array([-1.0]), np.array([2.0])]
f = np.array([0.0, 3.0, 0.1, 1.0])
F1 = f.reshape(-1, 1, 1)
F2 = np.zeros_like(F1)
print("three one-feature predictors with weights 1, -1, 2; four pairs with feature gap", f)
probs = np.array([sigmoid(F1[:, 0, 0] * w[0]) for w in ws])
for i in range(len(f)):
    print(f"pair {i}: member probabilities {probs[:, i].round(3)}, variance {probs[:, i].var():.4f}")
print("top 2 pairs to ask the human about:", select_queries(ws, F1, F2, 2).tolist())
print()
print("normalize [1, 2, 3]:", normalize_rewards(np.array([1.0, 2.0, 3.0])).round(4))
