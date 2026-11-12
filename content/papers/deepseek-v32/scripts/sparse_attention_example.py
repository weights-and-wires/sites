"""Attention over the selected keys, and when it equals dense attention.

Equation 2 (Section 2.1): the output of query t is attention between h_t and
only the entries whose index score is in the top-k. The toy below uses one
head, scalar scores written by hand, and the selections from
indexer_example.py for t = 3 (k = 3).

Run: python3 scripts/sparse_attention_example.py
"""

import numpy as np


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# Attention logits of query t = 3 against keys 0..3, and the values.
logits = np.array([0.0, 1.0, 2.0, 0.0])
V = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [-1.0, 1.0]])

dense_w = softmax(logits)
dense_out = dense_w @ V
print("dense weights  =", np.round(dense_w, 4).tolist())
print("dense output   =", np.round(dense_out, 4).tolist())

selected = [0, 1, 2]  # top-3 index scores for t = 3 (indexer_example.py)
sparse_w = softmax(logits[selected])
sparse_out = sparse_w @ V[selected]
print("sparse weights =", np.round(sparse_w, 4).tolist(), "over keys", selected)
print("sparse output  =", np.round(sparse_out, 4).tolist())
print("dense weight left on the dropped key 3 =", round(float(dense_w[3]), 4))
print("max abs output difference =", round(float(np.abs(dense_out - sparse_out).max()), 4))

# k >= number of visible keys keeps every key, so the outputs are identical.
all_w = softmax(logits[[0, 1, 2, 3]])
print("k = 4 equals dense:", bool(np.allclose(all_w @ V, dense_out)))

# k = 1 returns the value of the single selected key.
print("k = 1 output (key 2 only) =", (softmax(logits[[2]]) @ V[[2]]).tolist())
