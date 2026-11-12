import numpy as np
from submission import compatible_features, fit_compatible_critic, policy_gradient_with_critic


def softmax_rows(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def policy(phi, theta):
    return softmax_rows(phi @ theta)


def random_mdp(n_states, n_actions, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((n_states, n_actions, n_states)) ** 2
    P /= P.sum(axis=2, keepdims=True)
    return P, rng.normal(size=(n_states, n_actions))


def evaluate(P, R, pi, gamma, s0):
    """Return V, Q, d for the start-state formulation."""
    p_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, R)
    V = np.linalg.solve(np.eye(len(r_pi)) - gamma * p_pi, r_pi)
    Q = R + gamma * np.einsum("sat,t->sa", P, V)
    e0 = np.zeros(len(r_pi))
    e0[s0] = 1.0
    d = np.linalg.solve(np.eye(len(r_pi)) - gamma * p_pi.T, e0)
    return V, Q, d


def true_gradient_fd(P, R, phi, theta, gamma, s0, h=1e-6):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        up, dn = theta.copy(), theta.copy()
        up[i] += h
        dn[i] -= h
        j_up = evaluate(P, R, policy(phi, up), gamma, s0)[0][s0]
        j_dn = evaluate(P, R, policy(phi, dn), gamma, s0)[0][s0]
        g[i] = (j_up - j_dn) / (2 * h)
    return g


def page_example():
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 1] = P[1, 1, 0] = 1.0
    R = np.array([[0.0, 0.0], [1.0, 0.0]])
    phi = np.zeros((2, 2, 1))
    phi[:, 1, 0] = 1.0
    return P, R, phi


def test_features_by_hand():
    _, _, phi = page_example()
    pi = np.full((2, 2), 0.5)
    psi = compatible_features(phi, pi)
    assert np.allclose(psi[:, :, 0], [[-0.5, 0.5], [-0.5, 0.5]]), f"expected -0.5 and 0.5 in each state, got {psi[:, :, 0].tolist()}"


def test_features_are_grad_log_pi():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(3, 4, 2))
    theta = rng.normal(size=2)
    psi = compatible_features(phi, policy(phi, theta))
    h = 1e-6
    for k in range(2):
        up, dn = theta.copy(), theta.copy()
        up[k] += h
        dn[k] -= h
        fd = (np.log(policy(phi, up)) - np.log(policy(phi, dn))) / (2 * h)
        assert np.allclose(psi[:, :, k], fd, atol=1e-6), f"parameter {k}: psi must equal d log pi / d theta; max error {np.abs(psi[:, :, k] - fd).max()}"


def test_features_have_zero_policy_mean():
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(3, 4, 2))
    pi = policy(phi, rng.normal(size=2))
    psi = compatible_features(phi, pi)
    m = np.einsum("sa,sak->sk", pi, psi)
    assert np.allclose(m, 0, atol=1e-10), f"sum_a pi(s,a) psi(s,a) must be 0 in every state, got {m.tolist()}"


def test_page_example_numbers():
    P, R, phi = page_example()
    pi = np.full((2, 2), 0.5)
    psi = compatible_features(phi, pi)
    _, Q, d = evaluate(P, R, pi, 0.5, 0)
    w = fit_compatible_critic(psi, pi, d, Q)
    assert np.allclose(w, [-0.125]), f"expected w = [-0.125], got {w}"
    g = policy_gradient_with_critic(psi, pi, d, w)
    assert np.allclose(g, [-0.0625]), f"expected gradient [-0.0625], got {g}"


def test_critic_satisfies_equation_3():
    rng = np.random.default_rng(2)
    P, R = random_mdp(3, 3, seed=3)
    phi = rng.normal(size=(3, 3, 2))
    pi = policy(phi, np.array([0.4, -0.3]))
    _, Q, d = evaluate(P, R, pi, 0.9, 0)
    psi = compatible_features(phi, pi)
    w = fit_compatible_critic(psi, pi, d, Q)
    resid = np.einsum("sa,sak,sa->k", d[:, None] * pi, psi, Q - psi @ w)
    assert np.allclose(resid, 0, atol=1e-9), f"Equation 3 requires sum d pi psi (Q - f_w) = 0, got {resid}"
    assert np.abs(Q - psi @ w).max() > 0.1, "f_w should not equal Q here: 9 pairs and 2 parameters"


def test_gradient_with_critic_matches_true_gradient():
    for seed in range(3):
        rng = np.random.default_rng(30 + seed)
        P, R = random_mdp(3, 3, seed=40 + seed)
        phi = rng.normal(size=(3, 3, 2))
        theta = rng.normal(size=2)
        pi = policy(phi, theta)
        _, Q, d = evaluate(P, R, pi, 0.9, 0)
        psi = compatible_features(phi, pi)
        w = fit_compatible_critic(psi, pi, d, Q)
        got = policy_gradient_with_critic(psi, pi, d, w)
        want = true_gradient_fd(P, R, phi, theta, 0.9, 0)
        assert np.allclose(got, want, atol=1e-6), f"seed {seed}: gradient with f_w {got}, finite differences of J {want}"

