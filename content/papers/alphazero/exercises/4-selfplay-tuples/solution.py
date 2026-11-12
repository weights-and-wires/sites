import numpy as np

K = 3  # action a takes a + 1 stones
C_PUCT = 2.0


def visits_to_policy(N, tau):
    """pi proportional to N ** (1 / tau); tau == 0 is greedy with ties shared."""
    N = np.asarray(N, dtype=float)
    if tau == 0:
        best = N == N.max()
        return best / best.sum()
    with np.errstate(divide="ignore"):
        logits = np.log(N) / tau
    logits = logits - logits.max()
    w = np.exp(logits)
    return w / w.sum()


def search(n_root, prior_fn, value_fn, n_sims):
    """PUCT search from a pile of n_root stones. Returns the root visit counts, shape (3,).

    prior_fn(n) gives the prior over the 3 moves and value_fn(n) the value for the
    player to move. This function is deterministic: the same inputs give the same counts.
    """
    tree = {}

    def new_node(n):
        legal = np.arange(K) + 1 <= n
        p = np.asarray(prior_fn(n), dtype=float) * legal
        return {"N": np.zeros(K), "W": np.zeros(K), "P": p / p.sum(), "legal": legal}

    tree[n_root] = new_node(n_root)
    for _ in range(n_sims):
        s, path = n_root, []
        while s > 0 and s in tree:
            nd = tree[s]
            Q = np.divide(nd["W"], nd["N"], out=np.zeros(K), where=nd["N"] > 0)
            u = C_PUCT * nd["P"] * np.sqrt(nd["N"].sum()) / (1.0 + nd["N"])
            a = int(np.argmax(np.where(nd["legal"], Q + u, -np.inf)))
            path.append((s, a))
            s -= a + 1
        v = -1.0 if s == 0 else None
        if s > 0:
            tree[s] = new_node(s)
            v = value_fn(s)
        for state, a in reversed(path):
            v = -v
            tree[state]["N"][a] += 1
            tree[state]["W"][a] += v
    return tree[n_root]["N"]


def self_play_game(n_start, prior_fn, value_fn, n_sims, rng, tau_moves):
    """Play one game of Nim against itself and return the training tuples.

    At each state n, run search and let N be the root visit counts. Play the
    first `tau_moves` moves by sampling a move from visits_to_policy(N, 1.0)
    with rng.choice; play every later move greedily (argmax of N). The game
    ends when the pile is empty.

    Return a list with one tuple (n, pi, z) per move, in order. pi is always
    visits_to_policy(N, 1.0), also on greedy moves. z is the outcome for the
    player to move at n: +1 if that player takes the last stone, else -1.
    """
    history, n = [], n_start
    while n > 0:
        N = search(n, prior_fn, value_fn, n_sims)
        target = visits_to_policy(N, 1.0)
        if len(history) < tau_moves:
            a = int(rng.choice(K, p=target))
        else:
            a = int(np.argmax(N))
        history.append((n, target))
        n -= a + 1
    T = len(history)
    return [(s, pi, 1.0 if (T - 1 - i) % 2 == 0 else -1.0) for i, (s, pi) in enumerate(history)]
