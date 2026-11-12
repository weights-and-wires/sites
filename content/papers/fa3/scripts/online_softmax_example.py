"""Tiled attention with an online softmax, the consumer loop of Algorithm 1
(Section 3.1) with the two GEMMs written as plain matrix products.

Run: python3 scripts/online_softmax_example.py
"""

import numpy as np


def naive_attention(Q, K, V, tau=1.0):
    S = tau * Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P @ V


def tiled_attention(Q, K, V, Bc, tau=1.0, trace=False):
    """Return (O, L) for one query block; stream K and V in blocks of Bc rows."""
    Br = Q.shape[0]
    O = np.zeros((Br, V.shape[1]))
    l = np.zeros(Br)
    m = np.full(Br, -np.inf)
    for j in range(0, K.shape[0], Bc):
        S = tau * Q @ K[j:j + Bc].T                      # line 17
        m_new = np.maximum(m, S.max(axis=1))             # line 18
        P = np.exp(S - m_new[:, None])                   # line 19
        alpha = np.exp(m - m_new)
        l = alpha * l + P.sum(axis=1)                    # line 19
        O = alpha[:, None] * O + P @ V[j:j + Bc]         # line 21
        m = m_new
        if trace:
            print(f"  after block {j // Bc + 1}: m={m[0]:.4f} l={l[0]:.4f} "
                  f"alpha={alpha[0]:.4f} O~={np.round(O[0], 4)}")
    return O / l[:, None], m + np.log(l)                 # lines 24-25


print("One query row, scores [2, 0, 3, 1], two key blocks of 2, tau = 1")
Q = np.array([[1.0]])
K = np.array([[2.0], [0.0], [3.0], [1.0]])
V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
O, L = tiled_attention(Q, K, V, Bc=2, trace=True)
print(f"  final O = {np.round(O[0], 4)}  L = {L[0]:.4f}")
print(f"  naive O = {np.round(naive_attention(Q, K, V)[0], 4)}  "
      f"logsumexp = {np.log(np.exp([2, 0, 3, 1]).sum()):.4f}")

print()
print("Seeded random check: N=128 keys, d=64, one query block of 32 rows")
rng = np.random.default_rng(0)
Q = rng.normal(size=(32, 64))
K = rng.normal(size=(128, 64))
V = rng.normal(size=(128, 64))
tau = 1 / np.sqrt(64)
want = naive_attention(Q, K, V, tau)
for Bc in (128, 32, 7, 1):
    O, _ = tiled_attention(Q, K, V, Bc, tau)
    print(f"  Bc={Bc:<4} max |tiled - naive| = {np.abs(O - want).max():.2e}")

print()
print("Large scores: exp overflows in float64 above about 709 unless you subtract the max")
Q2 = rng.normal(size=(4, 64)) * 12
K2 = rng.normal(size=(64, 64)) * 12
V2 = rng.normal(size=(64, 8))
S2 = Q2 @ K2.T
print(f"  max score = {S2.max():.1f}")
with np.errstate(over="ignore", invalid="ignore"):
    P_bad = np.exp(S2)
    bad = (P_bad / P_bad.sum(axis=1, keepdims=True)) @ V2
print(f"  softmax without max subtraction -> finite? {bool(np.isfinite(bad).all())}")
O2, _ = tiled_attention(Q2, K2, V2, Bc=16)
print(f"  tiled with running max         -> max |tiled - naive| = "
      f"{np.abs(O2 - naive_attention(Q2, K2, V2)).max():.2e}")
