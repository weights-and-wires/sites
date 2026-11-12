"""Compare the RNN transducer lattice with CTC (Section 1 and Section 2.3).

Run: python3 scripts/ctc_compare.py
"""

import itertools
import math

import numpy as np

from rnnt_ref import count_alignments, emit_null, forward, lattice_probs


def ctc_collapse(path):
    out = []
    prev = None
    for s in path:
        if s != prev and s != 0:
            out.append(s)
        prev = s
    return tuple(out)


def ctc_paths(T, label, K):
    """Number of CTC paths (blank = 0, labels 1..K) that collapse to `label`."""
    return sum(1 for p in itertools.product(range(K + 1), repeat=T) if ctc_collapse(p) == label)


def ctc_prob(y, label):
    """p(label | x) by enumeration. y has shape (T, K+1) with blank at column 0."""
    T, K1 = y.shape
    return sum(
        math.prod(y[t, p[t]] for t in range(T))
        for p in itertools.product(range(K1), repeat=T)
        if ctc_collapse(p) == label
    )


print("Alignment counts for the target (a, b), K = 2 labels")
for T in (1, 2, 3, 4, 5):
    print(f"  T={T}: CTC paths = {ctc_paths(T, (1, 2), 2):3d}   RNN-T alignments = {count_alignments(T, 2) if T >= 1 else 0}")
print("A target with U = 3 labels and one frame: CTC paths =", ctc_paths(1, (1, 2, 1), 2),
      " RNN-T alignments =", count_alignments(1, 3))

T, U, K = 100, 30, 40
print(f"\nT={T}, U={U}, K={K}")
print(f"  CTC output tensor        T * (K+1)           = {T * (K + 1):,} values")
print(f"  RNN-T joint tensor       T * (U+1) * (K+1)   = {T * (U + 1) * (K + 1):,} values")
print(f"  ratio                    {(U + 1):d}x")
print(f"  RNN-T lattice nodes      T * (U+1)           = {T * (U + 1):,}")
print(f"  CTC forward-table cells  T * (2U+1)          = {T * (2 * U + 1):,}")
print(f"  RNN-T alignments         C(T+U-1, U)         = {count_alignments(T, U):.3e}")
print(f"  exponentials: per-node softmax {T * (U + 1) * (K + 1):,} vs exp(f), exp(g) once {(T + U + 1) * (K + 1):,}")

# Conditional independence. Two frames with the same acoustic evidence.
# Columns: null, a, b.
frame = np.array([2.0, 3.0, 3.0])
F = np.tile(frame, (2, 1))
ctc_frame = frame / frame.sum()
ctc_y = np.tile(ctc_frame, (2, 1))
print("\nTwo frames with identical evidence; per-frame distribution", ctc_frame)
print(f"  CTC:   p(ab|x) = {ctc_prob(ctc_y, (1, 2)):.5f}   p(ba|x) = {ctc_prob(ctc_y, (2, 1)):.5f}")


def g_weights(prefix):
    """A hand-made prediction network: after an `a`, the label `b` is four times as likely."""
    if prefix and prefix[-1] == 1:
        return np.array([1.0, 1.0, 4.0])
    return np.array([1.0, 1.0, 1.0])


def transducer_prob(F, y):
    G = np.stack([g_weights(tuple(y[:u])) for u in range(len(y) + 1)])
    probs = lattice_probs(np.log(F), np.log(G))
    emit, null = emit_null(probs, y)
    return forward(emit, null)[1]


pab, pba = transducer_prob(F, (1, 2)), transducer_prob(F, (2, 1))
print(f"  RNN-T: Pr(ab|x) = {pab:.5f}   Pr(ba|x) = {pba:.5f}   ratio = {pab / pba:.3f}")
