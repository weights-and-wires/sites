"""Numbers for the dueling-head and prioritized-replay examples.

Run: python3 scripts/dueling_priorities.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Scalar dueling aggregation (Section 2): q = v + a - mean(a).
v = 1.0
adv = np.array([0.5, -0.5, 3.0])
q = v + adv - adv.mean()
print("q            =", q)
print("mean(q) == v :", round(float(q.mean()), 6))
print("argmax q == argmax a:", int(q.argmax()) == int(adv.argmax()))

# Distributional dueling (Section 3): 2 actions, 3 atoms, softmax over atoms.
vd = np.array([0.0, 1.0, 0.0])
ad = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, 0.0]])
logits = vd[None, :] + ad - ad.mean(axis=0, keepdims=True)
e = np.exp(logits - logits.max(axis=1, keepdims=True))
p = e / e.sum(axis=1, keepdims=True)
print("logits =\n", logits)
print("p =\n", p)
print("row sums =", p.sum(axis=1))

# Prioritized replay: P(i) = p_i^w / sum p^w, weight = (N P)^-beta / max.
pri = np.array([0.04, 0.25, 1.0, 0.16])  # KL values
omega = 0.5
P = pri**omega / np.sum(pri**omega)
print("pri^0.5      =", pri**omega)
print("P            =", P)
N = len(pri)
for beta in (0.4, 1.0):
    w = (N * P) ** (-beta)
    print(f"beta={beta}: raw w =", w, " normalized =", w / w.max())
