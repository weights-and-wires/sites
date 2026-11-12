import numpy as np

K = 3  # action a takes a + 1 stones


def new_node(n, prior_fn):
    """Create the statistics for a state with n stones. Illegal moves get prior 0."""
    legal = np.arange(K) + 1 <= n
    p = np.asarray(prior_fn(n), dtype=float) * legal
    return {"N": np.zeros(K), "W": np.zeros(K), "P": p / p.sum(), "legal": legal}


def select(node, c_puct):
    """Return the legal action with the highest Q + u."""
    N, W = node["N"], node["W"]
    Q = np.divide(W, N, out=np.zeros(K), where=N > 0)
    u = c_puct * node["P"] * np.sqrt(N.sum()) / (1.0 + N)
    return int(np.argmax(np.where(node["legal"], Q + u, -np.inf)))


def mcts(n_root, prior_fn, value_fn, n_sims, c_puct=5.0):
    """Run n_sims simulations from n_root and return the root visit counts (shape (3,))."""
    tree = {n_root: new_node(n_root, prior_fn)}
    for _ in range(n_sims):
        s, path = n_root, []
        while s > 0 and s in tree:
            a = select(tree[s], c_puct)
            path.append((s, a))
            s -= a + 1
        if s == 0:
            v = -1.0
        else:
            tree[s] = new_node(s, prior_fn)
            v = value_fn(s)
        for state, a in reversed(path):
            v = -v
            tree[state]["N"][a] += 1
            tree[state]["W"][a] += v
    return tree[n_root]["N"]
