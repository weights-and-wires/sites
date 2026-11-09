import numpy as np


def residual_distribution(p, q):
    """Finished for you: norm(max(0, p - q))."""
    r = np.maximum(0.0, np.asarray(p, dtype=float) - np.asarray(q, dtype=float))
    return r / r.sum()


def verify_draft(p_dists, q_dists, draft, r, rng):
    """Return draft[:n] + [t] as a list of ints.

    p_dists: (gamma + 1, V) target distributions, one per position.
    q_dists: (gamma, V) draft distributions.
    draft:   (gamma,) drafted token IDs.
    r:       (gamma,) uniform draws; guess i is rejected when r[i] > p/q.
    """
    gamma = len(draft)
    n = gamma
    for i in range(gamma):
        x = draft[i]
        if r[i] > p_dists[i][x] / q_dists[i][x]:
            n = i
            break
    if n < gamma:
        dist = residual_distribution(p_dists[n], q_dists[n])
    else:
        dist = p_dists[gamma]
    t = int(rng.choice(len(dist), p=dist))
    return [int(x) for x in draft[:n]] + [t]
