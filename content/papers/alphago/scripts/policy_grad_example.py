"""Numbers for the "Three training stages" example.

Run: python3 scripts/policy_grad_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


x = np.zeros(3)
a = 1
p = softmax(x)
e = np.zeros(3)
e[a] = 1.0
grad = e - p  # d log p(a) / d logits
print("p            =", p)
print("SL gradient  =", grad)
for z, b in ((+1.0, 0.0), (-1.0, 0.0), (+1.0, 0.25)):
    print(f"RL gradient, z = {z:+.0f}, baseline = {b:.2f}: ", (z - b) * grad)

# One ascent step with step size 0.5 after a win and after a loss.
for z in (+1.0, -1.0):
    x2 = x + 0.5 * z * grad
    print(f"p(a) after a step with z = {z:+.0f}:", round(float(softmax(x2)[a]), 4), "(was", round(float(p[a]), 4), ")")
