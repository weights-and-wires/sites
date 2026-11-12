"""Composite reward (Section 3.1, Equation 6)."""

from azr_lib import composite_reward

print("solver, correct and well formatted     ", composite_reward(1.0, True, True))
print("solver, wrong but well formatted       ", composite_reward(0.0, True, False))
print("solver, formatting error               ", composite_reward(1.0, False, True))
print("proposer, valid task, learnability 0.75", composite_reward(0.75, True, True))
print("proposer, valid task, learnability 0   ", composite_reward(0.0, True, True))
print("proposer, invalid task                 ", composite_reward(0.75, True, False))
