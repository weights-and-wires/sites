"""Compute the numbers for the top-2 gating example.

Run: python3 scripts/top2_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


logits = np.array([2.0, 0.5, 1.0, -1.0, 0.0, 1.5, -0.5, 0.2])
order = np.argsort(-logits)
top2 = order[:2]
print("logits            =", logits.tolist())
print("top-2 experts     =", top2.tolist())

full = softmax(logits)
print("softmax over all  =", full)
print("  mass on top-2   =", round(float(full[top2].sum()), 4))

masked = np.full_like(logits, -np.inf)
masked[top2] = logits[top2]
g = softmax(masked)
print("Softmax(Top2)     =", g)
print("  sum             =", float(g.sum()))

renorm = full[top2] / full[top2].sum()
print("renormalized full softmax on top-2 =", renorm)
print("  same as Softmax(Top2)?", bool(np.allclose(renorm, g[top2])))

two = softmax(logits[top2])
print("sigmoid of gap 0.5 =", round(float(1 / (1 + np.exp(-0.5))), 4))
print("two-way softmax    =", two)

# Output mix with scalar stand-in expert outputs.
E = np.arange(1, 9) * 10.0
print("expert outputs     =", E.tolist())
print("y = sum G_i E_i    =", round(float(g @ E), 4))
print("y without renorm   =", round(float(full[top2] @ E[top2]), 4))
