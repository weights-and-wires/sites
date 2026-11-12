"""Overestimation of a max over noisy estimates, and the double-estimator fix.

Ten actions all have true value 0. Two independent value estimates are each
true value + N(0, 1) noise. The one-estimator target takes max_a of one
estimate. The double target picks the action with estimate A and reads its
value from estimate B (the pattern of Double DQN, Section 2).

Run: python3 scripts/overestimation.py
"""

import numpy as np

rng = np.random.default_rng(0)
trials, actions = 200_000, 10
qa = rng.normal(size=(trials, actions))
qb = rng.normal(size=(trials, actions))

single = qa.max(axis=1)
pick = qa.argmax(axis=1)
double = qb[np.arange(trials), pick]
print("true max value                     = 0.00")
print(f"mean of max_a qA (one estimator)   = {single.mean():.2f}")
print(f"mean of qB[argmax_a qA] (double)   = {double.mean():.2f}")
