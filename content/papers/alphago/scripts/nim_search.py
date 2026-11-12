"""Numbers for the "A small search on Nim" example.

A pile of n stones, players alternate, each takes 1, 2, or 3 stones, and the
player who takes the last stone wins. The player to move at n wins exactly when
n % 4 != 0. Action index a means "take a + 1 stones".

Run: python3 scripts/nim_search.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

K = 3
C_PUCT = 5.0  # Extended Data Table 5


def exact_value(n):
    """Value for the player to move: +1 if n % 4 != 0, else -1."""
    return -1.0 if n % 4 == 0 else 1.0


def flat_value(n):
    return 0.0


def uniform_prior(n):
    return np.ones(K) / K


def mcts(n_root, prior_fn, value_fn, n_sims, c_puct=C_PUCT):
    """Return the root visit counts N(root, a) after n_sims simulations."""
    tree = {}  # stone count -> {"N": ..., "W": ..., "P": ...}

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
            u = c_puct * nd["P"] * np.sqrt(nd["N"].sum()) / (1.0 + nd["N"])
            score = np.where(nd["legal"], Q + u, -np.inf)
            a = int(np.argmax(score))
            path.append((s, a))
            s -= a + 1
        if s == 0:
            v = -1.0  # the player to move has no stones to take: the opponent took the last one
        else:
            tree[s] = new_node(s)
            v = value_fn(s)
        # v is from the view of the player to move at the leaf; the edge into the
        # leaf was chosen by the other player, and signs alternate going up.
        for state, a in reversed(path):
            v = -v
            tree[state]["N"][a] += 1
            tree[state]["W"][a] += v
    return tree[n_root]["N"]


def best_move(counts):
    return int(np.argmax(counts)) + 1


if __name__ == "__main__":
    wrong_prior = lambda n: np.array([0.90, 0.05, 0.05])
    print("exact value net, uniform prior, 7 stones:")
    for sims in (10, 50, 200):
        c = mcts(7, uniform_prior, exact_value, sims)
        print(f"  {sims:4d} simulations: visits = {c}, take {best_move(c)}")
    print("exact value net, prior that favors taking 1, 7 stones:")
    for sims in (10, 50, 200):
        c = mcts(7, wrong_prior, exact_value, sims)
        print(f"  {sims:4d} simulations: visits = {c}, take {best_move(c)}")
    print("no value net (v = 0 everywhere except the terminal loss), uniform prior:")
    for n in (5, 6, 7):
        for sims in (50, 400, 2000):
            c = mcts(n, uniform_prior, flat_value, sims)
            print(f"  n = {n}, {sims:4d} simulations: visits = {c}, take {best_move(c)}")
