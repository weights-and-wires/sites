import numpy as np

N_STATES, GOAL, START = 7, 6, 3
R_MIN = -40


def step(s, a):
    """Corridor dynamics: a = 1 moves right, a = 0 moves left. Reward is -1."""
    s2 = min(s + 1, GOAL) if a == 1 else max(s - 1, 0)
    return s2, -1.0, s2 == GOAL


def fit_table(S, A, G):
    """Count (state, return-to-go, action) triples.

    Returns an array of shape (N_STATES, -R_MIN + 1, 2).
    """
    table = np.zeros((N_STATES, -R_MIN + 1, 2))
    g_idx = np.clip(np.asarray(G), R_MIN, 0).astype(int) - R_MIN
    np.add.at(table, (np.asarray(S), g_idx, np.asarray(A)), 1)
    return table


def action_probs(table, s, R):
    """P(a | s, R) from the counts, with [0.5, 0.5] when there is no data."""
    c = table[s, int(np.clip(R, R_MIN, 0)) - R_MIN]
    total = c.sum()
    if total == 0:
        return np.array([0.5, 0.5])
    return c / total


def rollout(table, target, rng, max_steps=200):
    """Finished for you. Samples actions from action_probs and decrements the target."""
    s, R, steps, done = START, float(target), 0, False
    while not done and steps < max_steps:
        a = int(rng.choice(2, p=action_probs(table, s, R)))
        s, r, done = step(s, a)
        R -= r
        steps += 1
    return steps, done
