"""Numbers for the "Visit counts as policy targets" example.

The search returns pi proportional to N(s, a)^(1 / tau) at the root. This
script turns one set of root visit counts into targets for several
temperatures and shows why the computation belongs in log space.

Run: python3 scripts/visit_targets.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def visits_to_policy(N, tau):
    """pi proportional to N^(1/tau). tau == 0 means greedy (ties share the mass)."""
    N = np.asarray(N, dtype=float)
    if tau == 0:
        best = N == N.max()
        return best / best.sum()
    with np.errstate(divide="ignore"):
        logits = np.log(N) / tau  # log 0 = -inf gives probability 0
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


if __name__ == "__main__":
    N = np.array([60, 30, 8, 2])
    print(f"root visit counts N = {N}, total {N.sum()}")
    for tau in (2.0, 1.0, 0.5, 0.25, 0.1, 0.0):
        pi = visits_to_policy(N, tau)
        print(f"tau = {tau:<5}: pi = {pi}  (sum {pi.sum():.4f})")

    with np.errstate(over="ignore", invalid="ignore"):
        naive = N.astype(float) ** (1 / 0.001)
        naive = naive / naive.sum()
    print(f"naive N^(1/0.001), normalized: {naive}")
    print(f"log-space tau = 0.001:         {visits_to_policy(N, 0.001)}")
    print(f"tied counts [5, 5, 0] at tau = 0: {visits_to_policy([5, 5, 0], 0.0)}")
