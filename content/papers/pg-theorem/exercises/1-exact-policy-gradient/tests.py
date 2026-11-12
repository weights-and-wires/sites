import numpy as np
from submission import action_values, discounted_state_weights, policy_gradient


def softmax_rows(theta):
    e = np.exp(theta - theta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def two_state_mdp():
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 1] = P[1, 1, 0] = 1.0
    R = np.array([[0.0, 0.0], [1.0, 0.0]])
    return P, R


def random_mdp(n_states, n_actions, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((n_states, n_actions, n_states)) ** 2
    P /= P.sum(axis=2, keepdims=True)
    return P, rng.normal(size=(n_states, n_actions))


def objective(P, R, theta, gamma, s0):
    """J = V(s0), computed by an independent route: the truncated sum of expected rewards."""
    pi = softmax_rows(theta)
    p_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, R)
    dist = np.zeros(P.shape[0])
    dist[s0] = 1.0
    total = 0.0
    for t in range(400):
        total += gamma**t * dist @ r_pi
        dist = dist @ p_pi
    return total


def finite_differences(P, R, theta, gamma, s0, h=1e-5):
    g = np.zeros_like(theta)
    for idx in np.ndindex(*theta.shape):
        up, dn = theta.copy(), theta.copy()
        up[idx] += h
        dn[idx] -= h
        g[idx] = (objective(P, R, up, gamma, s0) - objective(P, R, dn, gamma, s0)) / (2 * h)
    return g


def test_state_weights_by_hand():
    P, _ = two_state_mdp()
    pi = np.full((2, 2), 0.5)
    d = discounted_state_weights(P, pi, 0.5, 0)
    assert np.allclose(d, [1.5, 0.5]), f"expected d = [1.5, 0.5], got {d}"


def test_state_weights_sum_and_start():
    P, _ = random_mdp(4, 3, seed=1)
    pi = softmax_rows(np.random.default_rng(2).normal(size=(4, 3)))
    for gamma in (0.0, 0.5, 0.9):
        d = discounted_state_weights(P, pi, gamma, 2)
        assert abs(d.sum() - 1 / (1 - gamma)) < 1e-9, f"gamma={gamma}: expected sum {1 / (1 - gamma)}, got {d.sum()}"
    d0 = discounted_state_weights(P, pi, 0.0, 2)
    assert np.allclose(d0, [0, 0, 1, 0]), f"with gamma=0 all weight is on s0=2; expected [0, 0, 1, 0], got {d0}"


def test_state_weights_depend_on_start_state():
    P, _ = two_state_mdp()
    pi = np.full((2, 2), 0.5)
    d = discounted_state_weights(P, pi, 0.5, 1)
    assert np.allclose(d, [0.5, 1.5]), f"starting in state 1 expected [0.5, 1.5], got {d}"


def test_action_values_by_hand():
    P, R = two_state_mdp()
    q = action_values(P, R, np.full((2, 2), 0.5), 0.5)
    want = np.array([[0.125, 0.375], [1.375, 0.125]])
    assert np.allclose(q, want), f"expected {want.tolist()}, got {q.tolist()}"


def test_action_values_satisfy_bellman():
    P, R = random_mdp(4, 3, seed=3)
    pi = softmax_rows(np.random.default_rng(4).normal(size=(4, 3)))
    gamma = 0.9
    q = action_values(P, R, pi, gamma)
    v = (pi * q).sum(axis=1)
    want = R + gamma * np.einsum("sat,t->sa", P, v)
    assert np.allclose(q, want), f"Q must satisfy Q = R + gamma P V; max error {np.abs(q - want).max()}"


def test_gradient_by_hand():
    P, R = two_state_mdp()
    g = policy_gradient(P, R, np.zeros((2, 2)), 0.5, 0)
    want = np.array([[-0.09375, 0.09375], [0.15625, -0.15625]])
    assert np.allclose(g, want), f"expected {want.tolist()}, got {g.tolist()}"


def test_gradient_matches_finite_differences():
    for seed, (S, A, gamma) in enumerate([(3, 2, 0.9), (4, 3, 0.8), (2, 3, 0.95)]):
        P, R = random_mdp(S, A, seed=10 + seed)
        theta = np.random.default_rng(20 + seed).normal(size=(S, A))
        g = policy_gradient(P, R, theta, gamma, 0)
        fd = finite_differences(P, R, theta, gamma, 0)
        assert np.allclose(g, fd, atol=1e-6), f"seed {seed}: max error {np.abs(g - fd).max()}\nanalytic {g.tolist()}\nfinite differences {fd.tolist()}"


def test_gradient_uses_the_start_state():
    P, R = random_mdp(3, 2, seed=5)
    theta = np.random.default_rng(6).normal(size=(3, 2))
    g = policy_gradient(P, R, theta, 0.9, 2)
    fd = finite_differences(P, R, theta, 0.9, 2)
    assert np.allclose(g, fd, atol=1e-6), f"s0=2: max error {np.abs(g - fd).max()}"


def test_gradient_rows_sum_to_zero():
    P, R = random_mdp(3, 3, seed=7)
    theta = np.random.default_rng(8).normal(size=(3, 3))
    g = policy_gradient(P, R, theta, 0.9, 0)
    assert np.allclose(g.sum(axis=1), 0, atol=1e-10), f"softmax is invariant to a shift in a row; row sums should be 0, got {g.sum(axis=1)}"
