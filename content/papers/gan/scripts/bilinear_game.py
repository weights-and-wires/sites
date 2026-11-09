"""Simultaneous gradient descent-ascent on min_x max_y x*y spirals outward.

This is the smallest game where gradient play does not converge, even though
the saddle point (0, 0) exists. It illustrates why Section 3 stresses that D and
G must be kept in step.

Run: python3 scripts/bilinear_game.py
"""

import numpy as np

lr = 0.1
x, y = 1.0, 1.0
print("step | x | y | radius")
for step in range(0, 101):
    if step in (0, 1, 10, 50, 100):
        print(f"{step:4d} | {x:8.4f} | {y:8.4f} | {np.hypot(x, y):.4f}")
    x, y = x - lr * y, y + lr * x  # x descends on xy, y ascends on xy

print()
print("radius factor per step  =", round(float(np.sqrt(1 + lr**2)), 5))
print("radius after 100 steps  =", round(float(np.sqrt(2) * (1 + lr**2) ** 50), 4))
