"""Bidirectional and causal self-attention on a three-token example
(Section 3 and Figure 3). One head, d = 2, W_q = W_k = W_v = identity.

Run: python3 scripts/attention_masks.py
"""

import numpy as np


def attention(x, causal):
    d = x.shape[-1]
    scores = x @ x.T / np.sqrt(d)
    if causal:
        scores = np.where(np.tril(np.ones_like(scores)) == 1, scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ x, w


x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
for causal in (False, True):
    out, w = attention(x, causal)
    print(f"causal={causal}")
    print("  weights:\n", np.round(w, 4))
    print("  output row 0:", np.round(out[0], 4))

# Change the last token and see whether row 0 moves.
x2 = x.copy()
x2[2] = [-3.0, 2.0]
for causal in (False, True):
    a, _ = attention(x, causal)
    b, _ = attention(x2, causal)
    print(f"causal={causal}: change in output row 0 after editing token 3 = {np.abs(a[0] - b[0]).max():.4f}")
