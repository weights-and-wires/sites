#!/usr/bin/env python3
"""The InfoNCE loss of Equations 2 and 3 on hand-sized numbers.

Run from the paper folder: python3 scripts/infonce_example.py
"""

import numpy as np


def loss(cos_pos, cos_negs, tau):
    logits = np.array([cos_pos] + list(cos_negs)) / tau
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return -np.log(p[0]), p[0]


print("scale 1/tau at tau = 0.02:", 1 / 0.02)

# One query: cosine 0.80 to the positive, 0.78 to a hard negative,
# 0.70 to an in-batch negative.
cos_pos, cos_negs = 0.80, [0.78, 0.70]
for tau in (0.02, 1.0):
    l, p = loss(cos_pos, cos_negs, tau)
    print(f"tau={tau}: loss={l:.4f}, softmax share of the positive={p:.4f}")

# Only the hard negative, so loss = ln(1 + e^-1).
l, _ = loss(0.80, [0.78], 0.02)
print(f"one hard negative, tau=0.02: loss={l:.4f}")
print(f"ln(1 + exp(-1)) = {np.log(1 + np.exp(-1)):.4f}")

# Gap needed for a 0.01 loss against one negative at tau = 0.02.
gap = 0.02 * np.log(1 / (np.exp(0.01) - 1))
print(f"cosine gap for loss 0.01 against one negative at tau=0.02: {gap:.4f}")

# The self-check: the same one-negative gap of 0.02 at tau = 1.
print(f"one negative, cosines 0.80 and 0.78, tau=1: loss={np.log(1 + np.exp(-0.02)):.4f}")
print(f"ln 2 = {np.log(2):.4f}")
