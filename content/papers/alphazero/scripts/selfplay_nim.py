"""Numbers for the "Self-play produces the training data" example.

A pile of n stones, players alternate, each takes 1, 2, or 3 stones, and the
player who takes the last stone wins (action index a takes a + 1 stones). The
player to move at n wins exactly when n % 4 != 0.

Part 1 plays self-play games with a fixed prior and value function and prints
the (s, pi, z) training tuples. Part 2 closes the loop: a tabular "network"
(one row of logits and one value per pile size) starts at random, plays itself
with search, and trains on the tuples with the AlphaZero loss. No human data and
no rules of strategy enter, only the move rules.

Run: python3 scripts/selfplay_nim.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)
K = 3
C_PUCT = 2.0
EPS, ALPHA = 0.25, 0.5  # root noise: eps from the paper, alpha chosen for K = 3


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def search(n_root, prior_fn, value_fn, n_sims, rng=None, noise=False):
    """PUCT search. Returns root visit counts. Optional Dirichlet noise on root priors."""
    tree = {}

    def new_node(n, root=False):
        legal = np.arange(K) + 1 <= n
        p = np.asarray(prior_fn(n), dtype=float) * legal
        p = p / p.sum()
        if root and noise:
            eta = np.zeros(K)
            eta[legal] = rng.dirichlet(np.full(legal.sum(), ALPHA))
            p = (1 - EPS) * p + EPS * eta
        return {"N": np.zeros(K), "W": np.zeros(K), "P": p, "legal": legal}

    tree[n_root] = new_node(n_root, root=True)
    for _ in range(n_sims):
        s, path = n_root, []
        while s > 0 and s in tree:
            nd = tree[s]
            Q = np.divide(nd["W"], nd["N"], out=np.zeros(K), where=nd["N"] > 0)
            u = C_PUCT * nd["P"] * np.sqrt(nd["N"].sum()) / (1.0 + nd["N"])
            a = int(np.argmax(np.where(nd["legal"], Q + u, -np.inf)))
            path.append((s, a))
            s -= a + 1
        if s == 0:
            v = -1.0
        else:
            tree[s] = new_node(s)
            v = value_fn(s)
        for state, a in reversed(path):
            v = -v
            tree[state]["N"][a] += 1
            tree[state]["W"][a] += v
    return tree[n_root]["N"]


def visits_to_policy(N, tau):
    if tau == 0:
        best = N == N.max()
        return best / best.sum()
    with np.errstate(divide="ignore"):
        logits = np.log(N) / tau
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def self_play_game(n_start, prior_fn, value_fn, n_sims, rng, tau_moves=2, noise=False):
    """Play one game. Returns a list of (n, pi, z); z is for the player to move at n."""
    history, n = [], n_start
    while n > 0:
        N = search(n, prior_fn, value_fn, n_sims, rng, noise)
        tau = 1.0 if len(history) < tau_moves else 0.0
        pi = visits_to_policy(N, tau)
        history.append((n, visits_to_policy(N, 1.0)))  # the target always uses tau = 1
        a = int(rng.choice(K, p=pi))
        n -= a + 1
    # The player who made the last move won, so the last state has z = +1 for its mover.
    out = []
    for i, (state, target) in enumerate(history):
        z = 1.0 if (len(history) - 1 - i) % 2 == 0 else -1.0
        out.append((state, target, z))
    return out


def exact_value(n):
    return -1.0 if n % 4 == 0 else 1.0


def uniform_prior(n):
    return np.ones(K) / K


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    game = self_play_game(7, uniform_prior, exact_value, 100, rng, tau_moves=1)
    print("one self-play game from 7 stones, exact value, uniform prior, 100 simulations:")
    for n, pi, z in game:
        print(f"  n = {n}  pi = {pi}  z = {z:+.0f}")

    # Part 2: tabular network trained from scratch.
    NMAX = 15
    rng = np.random.default_rng(0)
    logits = 0.1 * rng.standard_normal((NMAX + 1, K))
    vals = np.zeros(NMAX + 1)

    def prior_fn(n):
        return softmax(logits[n])

    def value_fn(n):
        return float(vals[n])

    def report(it):
        ok = [int(np.argmax(np.where(np.arange(K) + 1 <= n, logits[n], -np.inf))) + 1 == n % 4
              for n in range(1, NMAX + 1) if n % 4]
        vs = [np.sign(vals[n]) == (1 if n % 4 else -1) for n in range(1, NMAX + 1)]
        print(f"  iteration {it:2d}: policy head alone picks the winning move in {sum(ok)} of {len(ok)} "
              f"winning piles; value head has the right sign in {sum(vs)} of {NMAX} piles")

    print("tabular self-play from 15 stones (random initial network, 12 games and 40 simulations per iteration):")
    report(0)
    lr, c = 0.5, 0.0
    for it in range(1, 31):
        data = []
        for _ in range(12):
            data += self_play_game(NMAX, prior_fn, value_fn, 40, rng, tau_moves=3, noise=True)
        for _ in range(5):  # a few gradient steps on the batch
            g_l, g_v = np.zeros_like(logits), np.zeros_like(vals)
            for n, pi, z in data:
                legal = np.arange(K) + 1 <= n
                p = np.where(legal, np.exp(logits[n] - logits[n][legal].max()), 0.0)
                p /= p.sum()
                g_l[n] += (p - pi) / len(data)  # d l / d logits
                g_v[n] += -2 * (z - vals[n]) / len(data)  # d l / d v
            logits -= lr * g_l
            vals -= lr * g_v
        if it in (1, 2, 5, 10, 20, 30):
            report(it)
