"""Numbers for "The problem": why discretizing the action space does not scale.

Run: python3 scripts/discretization.py
"""

# Introduction: a 7-degree-of-freedom arm with the coarsest grid a_i in {-k, 0, k}.
joints = 7
for levels in (3, 5, 10):
    print(f"{levels:>2} levels per joint, {joints} joints -> {levels**joints:,} discrete actions")

# A DQN-style network needs one output unit per action, so the output layer of a
# 256-unit hidden layer alone holds (256 + 1) * K weights.
hidden = 256
for levels in (3, 10):
    k = levels**joints
    print(f"output layer for {levels} levels: {(hidden + 1) * k:,} weights")

# The continuous alternative: one actor with N outputs, whatever the resolution.
print(f"actor output layer for {joints} joints: {(hidden + 1) * joints:,} weights")
