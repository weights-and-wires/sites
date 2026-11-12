"""Show how the upper clip bounds a probability increase, and how Clip-Higher
changes it (Section 3.1, Equation 10).

Run: python3 scripts/clip_higher_example.py
"""

import numpy as np

EPS_LOW, EPS_HIGH = 0.2, 0.28


def term(ratio, adv, eps_low, eps_high):
    """Per-token term min(r A, clip(r, 1 - eps_low, 1 + eps_high) A)."""
    clipped = np.clip(ratio, 1 - eps_low, 1 + eps_high)
    return np.minimum(ratio * adv, clipped * adv)


print("Upper bound on pi_theta for a token with A > 0: pi_old * (1 + eps_high)")
print("pi_old   eps=0.20 (bound)   eps_high=0.28 (bound)   gain")
for p in (0.01, 0.1, 0.5, 0.9):
    a, b = p * (1 + 0.2), p * (1 + EPS_HIGH)
    print(f"{p:<7}  {a:<17.4f}  {b:<21.4f}   {b - a:+.4f}")
print("A probability cannot exceed 1, so the 0.9 bounds of 1.08 and 1.152 impose no limit.")

print()
print("A token with pi_old = 0.01 and A = +1 after an update to pi_theta = 0.0125 (ratio 1.25)")
r = 0.0125 / 0.01
for name, eh in (("eps_high = 0.20", 0.2), ("eps_high = 0.28", 0.28)):
    t = float(term(r, 1.0, 0.2, eh))
    active = r <= 1 + eh
    print(f"{name}: term = {t:.4f}, gradient flows = {active}")

print()
print("Lower side, A = -1 and ratio 0.75 (the token lost 25% of its probability)")
for name, el in (("eps_low = 0.20", 0.2), ("eps_low = 0.28", 0.28)):
    t = float(term(0.75, -1.0, el, 0.28))
    active = 0.75 >= 1 - el
    print(f"{name}: term = {t:.4f}, gradient flows = {active}")
