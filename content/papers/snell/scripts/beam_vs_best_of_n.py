"""Beam search against best-of-N at an equal generation budget N.

This is a toy model of this page, not the paper's experiment. A solution has
K steps. Each step is good with probability Q, and the solution is correct only
if every step is good, so a bad prefix never recovers. The PRM scores a prefix
as 1 (all steps so far good) or 0, plus Gaussian noise of scale SIGMA. As in
Section 5.2, the budget N counts final answers: best-of-N samples N full
solutions, and beam search keeps N beams and a beam width M.

Run: python3 scripts/beam_vs_best_of_n.py
"""

import numpy as np

Q = 0.75
K = 4
SIGMA = 0.8
TRIALS = 20000
BUDGETS = [4, 16, 64, 256]
rng = np.random.default_rng(0)
ROWS = np.arange(TRIALS)[:, None]


def noisy(good):
    return good + SIGMA * rng.standard_normal(good.shape)


def best_of_n(n):
    good = rng.random((TRIALS, n, K)) < Q
    full_good = good.all(axis=2)
    pick = noisy(full_good.astype(float)).argmax(axis=1)
    return float(full_good[np.arange(TRIALS), pick].mean())


def beam_search(n, m):
    good = rng.random((TRIALS, n)) < Q                      # N first steps
    for _ in range(K - 1):
        scores = noisy(good.astype(float))
        top = np.argsort(-scores, axis=1)[:, : n // m]      # keep top N/M
        parents = good[ROWS, top]                           # (T, N/M)
        children = rng.random((TRIALS, n // m, m)) < Q      # M next steps each
        good = (parents[:, :, None] & children).reshape(TRIALS, n)
    pick = noisy(good.astype(float)).argmax(axis=1)
    return float(good[np.arange(TRIALS), pick].mean())


print(f"q={Q}, K={K}, sigma={SIGMA}, trials={TRIALS}, seed 0")
print(f"one sample is correct with probability q^K = {Q**K:.4f}")
print("   N   best-of-N  beam M=4  beam M=sqrt(N)")
for n in BUDGETS:
    m_sqrt = int(round(n ** 0.5))
    print(f"{n:4d}   {best_of_n(n):8.3f}  {beam_search(n, 4):8.3f}  {beam_search(n, m_sqrt):8.3f}")
