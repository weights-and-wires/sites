#!/usr/bin/env python3
"""Numbers for the Bradley-Terry preference model (Section 2.2.3, Equation 1)."""

import numpy as np
from pref_lib import pref_loss, pref_prob

print("segment 1 return 3.0, segment 2 return 1.0")
p = float(pref_prob(3.0, 1.0))
print(f"P[s1 > s2] = {p:.4f}; P[s2 > s1] = {1 - p:.4f}")
print(f"loss if the human prefers s1 (mu=1)   : {pref_loss(3.0, 1.0, 1.0):.4f}")
print(f"loss if the human prefers s2 (mu=0)   : {pref_loss(3.0, 1.0, 0.0):.4f}")
print(f"loss if the human says tie   (mu=0.5) : {pref_loss(3.0, 1.0, 0.5):.4f}")
print()
print("equal returns (5.0 and 5.0): P =", float(pref_prob(5.0, 5.0)))
print("shift both returns by +100: P =", round(float(pref_prob(103.0, 101.0)), 4), "(only the difference matters)")
print()
print("10% uniform-error adjustment (eps = 0.1):")
print(f"P[s1 > s2] = {float(pref_prob(3.0, 1.0, 0.1)):.4f}")
print("return gap 50, the human prefers the other segment (mu=0):")
print(f"loss, eps=0   : {pref_loss(50.0, 0.0, 0.0):.4f}")
print(f"loss, eps=0.1 : {pref_loss(50.0, 0.0, 0.0, 0.1):.4f}")
print(f"cap -log(eps/2) = {-np.log(0.05):.4f}")
