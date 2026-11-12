"""Worked example B: a flat group-relative advantage with a time cap."""

import numpy as np

returns = np.array([3.0, 5.0, 5.0, 9.0])  # tests passed when the cap hits
mean, std = returns.mean(), returns.std()  # population std
adv = (returns - mean) / (std + 1e-8)
print("returns   ", returns)
print("mean, std ", mean, round(float(std), 4))
print("variance  ", float(returns.var()))
print("advantages", np.round(adv, 2))
print("sum       ", round(float(adv.sum()), 6) + 0.0)
