"""Checks Equation 4 of the paper: block-wise attention equals dense attention.

Run from the paper folder: python3 scripts/paged_attention_example.py

The keys and values live in a pool of physical blocks. A block table maps the
logical blocks of one sequence to scattered physical blocks. The paper's
Figure 5 uses 3 logical blocks of 4 tokens; this script uses 10 tokens, so the
last block holds 2 tokens.
"""
import numpy as np

rng = np.random.default_rng(0)
d, B, n = 8, 4, 10
q = rng.normal(size=d)
K = rng.normal(size=(n, d))
V = rng.normal(size=(n, d))

# Dense attention, Equation 3: a_j = exp(q.k_j / sqrt d) / sum_t exp(q.k_t / sqrt d)
scores = K @ q / np.sqrt(d)
a = np.exp(scores - scores.max())
a /= a.sum()
dense = a @ V

# Physical pool of 6 blocks; sequence uses physical blocks 4, 0, 5 in that order.
pool_K = np.zeros((6, B, d))
pool_V = np.zeros((6, B, d))
table = [4, 0, 5]
for tok in range(n):
    p, off = table[tok // B], tok % B
    pool_K[p, off], pool_V[p, off] = K[tok], V[tok]

# Equation 4: scores per block, one softmax over all blocks, weighted sum of V_j with A_j.
filled = [min(B, n - j * B) for j in range(len(table))]
block_scores = [pool_K[p, :f] @ q / np.sqrt(d) for p, f in zip(table, filled)]
m = max(s.max() for s in block_scores)
exps = [np.exp(s - m) for s in block_scores]
denom = sum(e.sum() for e in exps)
paged = sum((e / denom) @ pool_V[p, :f] for e, p, f in zip(exps, table, filled))

print("block table:", table, "filled:", filled)
print("max |paged - dense| =", f"{np.abs(paged - dense).max():.2e}")

# Two bugs the exercise tests catch.
wrong_local = sum((e / e.sum()) @ pool_V[p, :f] for e, p, f in zip(exps, table, filled))
print("softmax per block instead of over all blocks: max error", f"{np.abs(wrong_local - dense).max():.3f}")
all_slots = np.concatenate([pool_K[p] @ q / np.sqrt(d) for p in table])
w = np.exp(all_slots - all_slots.max())
w /= w.sum()
wrong_pad = w @ np.concatenate([pool_V[p] for p in table])
print("attending to the 2 unfilled slots (zeros): max error", f"{np.abs(wrong_pad - dense).max():.3f}")
