"""The masked-LM loss is cross-entropy averaged over the selected positions
only (Section 3.1, Appendix C.2).

Run: python3 scripts/mlm_loss.py
"""

import numpy as np


def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


# 4 positions, vocabulary of 3. Positions 1 and 3 are selected.
logits = np.array(
    [
        [2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 1.0, 2.0],
    ]
)
labels = np.array([-100, 1, -100, 0])
lp = log_softmax(logits)
print("log-probabilities:\n", np.round(lp, 4))
sel = labels != -100
nll = -lp[sel, labels[sel]]
print(f"negative log-probability at the selected positions: {np.round(nll, 4)}")
print(f"MLM loss (mean over {sel.sum()} selected positions): {nll.mean():.4f}")

# What goes wrong if you average over all 4 positions with a dummy label 0.
dummy = np.where(sel, labels, 0)
nll_all = -lp[np.arange(4), dummy]
print(f"wrong: mean over all positions with dummy label 0: {nll_all.mean():.4f}")
print(f"wrong: sum over selected positions: {nll.sum():.4f}")

# Signal count: how many positions give a gradient.
T = 512
print(f"\nPositions with a training signal per {T}-token sequence: causal LM {T - 1}, MLM {round(0.15 * T)} (15 percent)")
