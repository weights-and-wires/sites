"""The indexer's KL loss in the dense warm-up and the sparse stage.

Equation 3 (dense warm-up): L = sum_t KL(p_t || softmax(I_t)), where p_t is
the main attention weights summed over heads and L1-normalized.
Equation 4 (sparse stage): the same KL, restricted to the selected set S_t.
The paper does not say whether the restricted distributions are renormalized
over S_t. This script renormalizes both, which is one reading of Equation 4.

Run: python3 scripts/kl_warmup_example.py
"""

import numpy as np


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


# Query t = 3 sees keys 0..3. Two main-attention heads give these weights.
head_a = np.array([0.1, 0.2, 0.6, 0.1])
head_b = np.array([0.3, 0.1, 0.4, 0.2])
summed = head_a + head_b
p = summed / summed.sum()  # L1 normalization
print("summed over heads =", np.round(summed, 4).tolist())
print("target p          =", np.round(p, 4).tolist())

I = np.array([1.0, 0.5, 1.5, 0.0])  # index scores for t = 3
q = softmax(I)
print("softmax(I)        =", np.round(q, 4).tolist())
print("dense KL (Eq 3)   =", round(kl(p, q), 4))
print("KL with I = log p =", round(kl(p, softmax(np.log(p))), 6))

S = [0, 1, 2]
pS = p[S] / p[S].sum()
qS = softmax(I[S])
print("p mass inside S   =", round(float(p[S].sum()), 4))
print("sparse KL (Eq 4)  =", round(kl(pS, qS), 4))
