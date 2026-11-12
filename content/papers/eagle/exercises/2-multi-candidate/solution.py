import numpy as np


def residual_distribution(p, q):
    """Finished for you: norm(max(0, p - q))."""
    r = np.maximum(0.0, np.asarray(p, dtype=float) - np.asarray(q, dtype=float))
    return r / r.sum()


def verify_node(p, q, cands, u, rng):
    """Return (token, i). i is the index of the accepted candidate, or -1 if
    every candidate was rejected and the token came from the adjusted target.

    p: (V,) target distribution.  q: (V,) draft distribution.
    cands: k candidate token IDs drawn i.i.d. from q.
    u: k uniform draws; candidate i is accepted when u[i] <= p_cur[x] / q[x].
    """
    cur = np.asarray(p, dtype=float)
    for i, (x, ui) in enumerate(zip(cands, u)):
        x = int(x)
        if ui <= cur[x] / q[x]:
            return x, i
        cur = residual_distribution(cur, q)
    return int(rng.choice(len(cur), p=cur)), -1
