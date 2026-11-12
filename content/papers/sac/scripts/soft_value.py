"""Numbers for the "Soft value" example (Section 3.2 Equation 1, Section 4.1 Equations 3 and 4).

The paper sets the temperature to 1 and folds it into the reward scale. This script keeps
alpha explicit so that you can see the equivalence.

Run: python3 scripts/soft_value.py
"""

import numpy as np


def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


q = np.array([1.0, 2.0, 3.0])

# 1. alpha = 1: the Boltzmann policy pi ~ exp(Q) and the soft value V = log sum exp Q.
alpha = 1.0
pi = np.exp(q / alpha - logsumexp(q / alpha))
entropy = -np.sum(pi * np.log(pi))
v_expect = np.sum(pi * (q - alpha * np.log(pi)))
v_closed = alpha * logsumexp(q / alpha)
print(f"Q = {q}")
print(f"alpha = 1: pi = {np.round(pi, 4)}  entropy = {entropy:.4f}")
print(f"  V = E[Q - log pi] = {v_expect:.4f}   closed form logsumexp(Q) = {v_closed:.4f}")

# 2. Other policies on the same Q, scored by the same V formula.
greedy = np.array([0.0, 0.0, 1.0])
uniform = np.ones(3) / 3
for name, p in (("greedy", greedy), ("uniform", uniform), ("softmax", pi)):
    nz = p > 0
    h = -np.sum(p[nz] * np.log(p[nz]))
    print(f"  {name:8s} E[Q] = {np.sum(p * q):.4f}  entropy = {h:.4f}  V = {np.sum(p * q) + h:.4f}")

# 3. Temperature: alpha -> 0 recovers max Q, large alpha approaches mean Q + alpha log 3.
for alpha in (0.1, 1.0, 10.0):
    print(f"alpha = {alpha:>4}: V = {alpha * logsumexp(q / alpha):.4f}")
print(f"  max Q = {q.max():.4f}; mean Q + 10 log 3 = {q.mean() + 10 * np.log(3):.4f}")

# 4. Two actions with equal Q: the soft value is Q + alpha log 2, because the best policy is uniform.
print(f"Q = (0, 0), alpha = 1: V = {logsumexp(np.array([0.0, 0.0])):.4f}  (log 2 = {np.log(2):.4f})")
