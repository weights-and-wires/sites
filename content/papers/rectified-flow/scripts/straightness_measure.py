#!/usr/bin/env python3
"""Hand-checkable values of the straightness S(Z) (Section 2.2, Equation 3).

    S(Z) = integral over t in [0, 1] of E |(Z1 - Z0) - dZ_t/dt|^2

A path is straight in the paper's sense only if it moves along the chord at
constant speed. Three single paths from Z0 = (0, 0):

* constant speed along the chord to (3, 4): S = 0;
* the same chord at speed that grows with time, Z_t = t^2 (3, 4): S = |chord|^2 / 3 = 25 / 3;
* a half circle from (0, 0) to (2, 0): S = pi^2 - 4.

Each is also computed by the discrete formula that Exercise 2 asks for.

Run from the repository root:

    python3 content/papers/rectified-flow/scripts/straightness_measure.py
"""

import math

import numpy as np


def straightness(path):
    """Discrete S for a path of shape (T + 1, n, d) on a uniform grid of T steps."""
    steps = len(path) - 1
    h = 1.0 / steps
    chord = path[-1] - path[0]
    rate = (path[1:] - path[:-1]) / h
    return float(h * ((chord[None] - rate) ** 2).sum(-1).mean(1).sum())


def main():
    t = np.linspace(0.0, 1.0, 1001)[:, None, None]
    chord = np.array([[[3.0, 4.0]]])

    constant = t * chord
    accel = t**2 * chord
    arc = np.concatenate([1.0 - np.cos(math.pi * t), np.sin(math.pi * t)], axis=-1)

    print(f"constant speed along the chord : S = {straightness(constant):.4f}   (exact 0)")
    print(f"Z_t = t^2 (3, 4)               : S = {straightness(accel):.4f}   (exact 25/3 = {25 / 3:.4f})")
    print(f"half circle from (0,0) to (2,0): S = {straightness(arc):.4f}   (exact pi^2 - 4 = {math.pi**2 - 4:.4f})")


if __name__ == "__main__":
    main()
