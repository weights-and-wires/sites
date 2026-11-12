"""Numbers for the "Delayed updates" example (Section 5.1, Equations 11-12; Section 5.2).

Run: python3 scripts/accumulating_error.py
"""

# A constant residual TD error delta shifts the value by delta * sum(gamma^k) (Equation 12).
delta = 0.1
for gamma in (0.9, 0.99):
    horizon = 1000
    total = sum(gamma**k * delta for k in range(horizon))
    print(f"gamma {gamma}: shift after {horizon} steps = {total:.3f}, limit delta/(1-gamma) = {delta / (1 - gamma):.3f}")

# Delay d = 2, tau = 0.005 (Section 6.1): counts over 1 million time steps.
T, d, tau = 1_000_000, 2, 0.005
print("critic updates:", T)
print("actor and target updates:", T // d)

steps = [t for t in range(1, 11) if t % d == 0]
print("actor steps among the first 10 (d = 2):", steps)

# Gap to the source after 1000 critic steps: targets move only on actor steps.
gap = (1 - tau) ** (1000 // d)
every = (1 - tau) ** 1000
print(f"gap left after 1000 critic steps: delayed {gap:.4f}, undelayed {every:.4f}")

# Half the gap: how many actor updates, and how many critic steps with d = 2?
import math

k = math.ceil(math.log(0.5) / math.log(1 - tau))
print(f"tau = {tau}: {k} target updates close half the gap, which is {k * d} critic steps at d = {d}")
