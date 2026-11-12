"""Numbers for the "Soft target updates" example (Section 3, theta' <- tau theta + (1 - tau) theta').

Run: python3 scripts/soft_update.py
"""

import math

import numpy as np

# 1. A target that tracks a frozen source. Source = 1, target starts at 0, so after
#    n updates the gap to the source is exactly (1 - tau)^n.
tau = 0.1
target = 0.0
print("tau = 0.1, source fixed at 1:")
for n in range(1, 4):
    target = tau * 1.0 + (1 - tau) * target
    print(f"  after {n} update(s): target = {target:.3f}, gap = {1 - target:.3f}")

# 2. Paper value tau = 0.001 (Section 7): how many updates until the gap shrinks?
tau = 0.001
for frac in (0.5, 0.1, 0.01):
    n = math.ceil(math.log(frac) / math.log(1 - tau))
    print(f"tau = 0.001: gap <= {frac:g} of the start after {n} updates")

# 3. A source that moves at speed v per step. The target settles to a constant
#    lag of v * (1 - tau) / tau, which is about v / tau.
v = 1e-3
source = 0.0
target = 0.0
for _ in range(20000):
    source += v
    target = tau * source + (1 - tau) * target
print(f"tau = 0.001, source speed {v}: lag after 20000 steps = {source - target:.6f}")
print(f"  predicted v * (1 - tau) / tau = {v * (1 - tau) / tau:.6f}")
print(f"  lag in steps = {(source - target) / v:.1f}")
