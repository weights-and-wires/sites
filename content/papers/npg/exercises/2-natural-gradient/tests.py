import numpy as np
from submission import fisher, natural_gradient, policy_gradient, probs


def make_problem():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(3, 3, 3))
    theta = rng.normal(size=3)
    rho = np.array([0.5, 0.3, 0.2])
    Q = rng.normal(size=(3, 3))
    return phi, theta, rho, Q


def test_policy_gradient_matches_finite_differences():
    phi, theta, rho, Q = make_problem()

    def J(t):
        return float(np.sum(rho[:, None] * probs(phi, t) * Q))

    want = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1e-6
        want[i] = (J(theta + e) - J(theta - e)) / 2e-6
    got = policy_gradient(phi, theta, rho, Q)
    assert np.allclose(got, want, atol=1e-6), f"expected {want}, got {got}"


def test_policy_gradient_value_from_the_page():
    phi, theta, rho, Q = make_problem()
    got = policy_gradient(phi, theta, rho, Q)
    want = np.array([0.0529, -0.0643, -0.0500])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {np.round(got, 4)}"


def test_policy_gradient_ignores_a_per_state_constant_in_q():
    phi, theta, rho, Q = make_problem()
    g = policy_gradient(phi, theta, rho, Q)
    g2 = policy_gradient(phi, theta, rho, Q + np.array([10.0, -3.0, 7.0])[:, None])
    assert np.allclose(g, g2), f"expected the same gradient after adding a constant to each state's Q, got {g2} vs {g}"


def test_natural_gradient_solves_the_damped_system():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(4, 4))
    F = B @ B.T + 0.1 * np.eye(4)
    g = rng.normal(size=4)
    for lam in (0.0, 0.5, 3.0):
        want = np.linalg.solve(F + lam * np.eye(4), g)
        got = natural_gradient(F, g, lam)
        assert np.allclose(got, want), f"lam = {lam}: expected {want}, got {got}"


def test_default_damping_is_one_thousandth():
    F = np.diag([1.0, 2.0])
    g = np.array([1.0, 1.0])
    want = g / (np.diag(F) + 1e-3)
    got = natural_gradient(F, g)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_damping_makes_the_singular_tabular_fisher_solvable():
    p = np.array([0.5, 0.3, 0.2])
    F = np.diag(p) - np.outer(p, p)
    g = np.array([0.2, -0.18, -0.02])
    got = natural_gradient(F, g, 1e-6)
    want = np.array([0.5, -0.5, 0.0])
    assert np.allclose(got, want, atol=1e-3), f"expected about {want} (Q - mean(Q)), got {got}"
    f_values = (np.eye(3) - p) @ got
    want_f = np.array([0.4, -0.6, -0.1])
    assert np.allclose(f_values, want_f, atol=1e-3), f"expected f(a) = w . psi(a) = {want_f} (the advantage), got {f_values}"


def test_inputs_are_not_modified():
    F = np.array([[2.0, 0.5], [0.5, 1.0]])
    g = np.array([1.0, -1.0])
    F0, g0 = F.copy(), g.copy()
    natural_gradient(F, g, 0.1)
    assert np.array_equal(F, F0), f"expected F unchanged, got {F}"
    assert np.array_equal(g, g0), f"expected g unchanged, got {g}"


def test_natural_direction_beats_the_gradient_at_equal_kl():
    phi, theta, rho, Q = make_problem()
    F = fisher(phi, theta, rho)
    g = policy_gradient(phi, theta, rho, Q)
    delta = 0.01

    def scale(d):
        return d * np.sqrt(2 * delta / (d @ F @ d))  # puts 0.5 d^T F d = delta

    gain_nat = g @ scale(natural_gradient(F, g, 0.0))
    gain_plain = g @ scale(g)
    assert gain_nat > gain_plain + 1e-6, f"expected the natural direction to gain more at the same quadratic KL, got {gain_nat} vs {gain_plain}"
