"""Show that dropping the tanh in the alignment model makes the weights ignore the decoder state.

Run: python3 scripts/linear_collapse.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)
n, n_prime, Tx = 4, 6, 5
Wa = rng.normal(size=(n_prime, n))
Ua = rng.normal(size=(n_prime, 2 * n))
va = rng.normal(size=n_prime)
H = rng.normal(size=(Tx, 2 * n))
states = [rng.normal(size=n) for _ in range(3)]


def softmax(e):
    e = np.exp(e - e.max())
    return e / e.sum()


print("with tanh (the paper's model):")
for s in states:
    print("  alpha =", softmax(np.tanh(Wa @ s + H @ Ua.T) @ va))
print("without tanh:")
for s in states:
    print("  alpha =", softmax((Wa @ s + H @ Ua.T) @ va))
