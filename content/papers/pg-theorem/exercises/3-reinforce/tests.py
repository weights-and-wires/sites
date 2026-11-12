import numpy as np
from submission import reinforce_gradient


def softmax_rows(theta):
    e = np.exp(theta - theta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def reference(states, actions, rewards, theta, gamma):
    """Slow, explicit loops."""
    pi = softmax_rows(theta)
    n, h = states.shape
    out = np.zeros_like(theta, dtype=float)
    for i in range(n):
        for t in range(h):
            g = sum(gamma**k * rewards[i, t + k] for k in range(h - t))
            score = np.zeros_like(out)
            score[states[i, t]] = -pi[states[i, t]]
            score[states[i, t], actions[i, t]] += 1.0
            out += gamma**t * g * score
    return out / n


def test_single_episode_by_hand():
    theta = np.zeros((2, 2))
    states = np.array([[0, 1]])
    actions = np.array([[1, 0]])
    rewards = np.array([[1.0, 2.0]])
    got = reinforce_gradient(states, actions, rewards, theta, 0.5)
    # G_1 = 2, G_0 = 1 + 0.5 * 2 = 2. Step 0: 1 * 2 * [-0.5, +0.5] in row 0; step 1: 0.5 * 2 * [+0.5, -0.5] in row 1.
    want = np.array([[-1.0, 1.0], [0.5, -0.5]])
    assert np.allclose(got, want), f"expected {want.tolist()}, got {got.tolist()}"


def test_matches_explicit_loops():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(3, 3))
    n, h = 6, 5
    states = rng.integers(0, 3, size=(n, h))
    actions = rng.integers(0, 3, size=(n, h))
    rewards = rng.normal(size=(n, h))
    for gamma in (0.0, 0.7, 1.0):
        got = reinforce_gradient(states, actions, rewards, theta, gamma)
        want = reference(states, actions, rewards, theta, gamma)
        assert np.allclose(got, want), f"gamma={gamma}: expected {want.tolist()}, got {got.tolist()}"


def test_average_over_episodes():
    rng = np.random.default_rng(1)
    theta = rng.normal(size=(2, 2))
    states = rng.integers(0, 2, size=(4, 3))
    actions = rng.integers(0, 2, size=(4, 3))
    rewards = rng.normal(size=(4, 3))
    whole = reinforce_gradient(states, actions, rewards, theta, 0.9)
    halves = 0.5 * (reinforce_gradient(states[:2], actions[:2], rewards[:2], theta, 0.9)
                    + reinforce_gradient(states[2:], actions[2:], rewards[2:], theta, 0.9))
    assert np.allclose(whole, halves), f"the estimate is a mean over episodes; expected {halves.tolist()}, got {whole.tolist()}"


# --- Statistical check against the exact gradient (Theorem 1) ---

def random_mdp(n_states, n_actions, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((n_states, n_actions, n_states)) ** 2
    P /= P.sum(axis=2, keepdims=True)
    return P, rng.normal(size=(n_states, n_actions))


def exact_gradient(P, R, theta, gamma, s0):
    pi = softmax_rows(theta)
    p_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, R)
    V = np.linalg.solve(np.eye(len(r_pi)) - gamma * p_pi, r_pi)
    Q = R + gamma * np.einsum("sat,t->sa", P, V)
    e0 = np.zeros(len(r_pi))
    e0[s0] = 1.0
    d = np.linalg.solve(np.eye(len(r_pi)) - gamma * p_pi.T, e0)
    return d[:, None] * pi * (Q - V[:, None])


def sample_episodes(P, R, theta, s0, n, h, seed):
    rng = np.random.default_rng(seed)
    pi_cum = np.cumsum(softmax_rows(theta), axis=1)
    p_cum = np.cumsum(P, axis=2)
    states = np.zeros((n, h), dtype=int)
    actions = np.zeros((n, h), dtype=int)
    s = np.full(n, s0)
    for t in range(h):
        states[:, t] = s
        a = (rng.random(n)[:, None] > pi_cum[s]).sum(axis=1)
        actions[:, t] = a
        s = np.minimum((rng.random(n)[:, None] > p_cum[s, a]).sum(axis=1), P.shape[0] - 1)
    return states, actions, R[states, actions]


def test_close_to_exact_gradient():
    P, R = random_mdp(3, 2, seed=0)
    theta = np.random.default_rng(1).normal(size=(3, 2))
    gamma = 0.9
    exact = exact_gradient(P, R, theta, gamma, 0)
    states, actions, rewards = sample_episodes(P, R, theta, 0, 5000, 50, seed=0)
    got = reinforce_gradient(states, actions, rewards, theta, gamma)
    err = np.abs(got - exact).max()
    assert err < 0.08, f"max abs error vs the exact gradient is {err:.4f}, expected below 0.08\nexact {exact.round(3).tolist()}\ngot   {got.round(3).tolist()}"
