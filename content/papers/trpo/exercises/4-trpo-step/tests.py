import numpy as np
from submission import conjugate_gradient, fisher_vector_product, probs, trpo_step

rng = np.random.default_rng(0)
N, K, D = 6, 3, 3
PHI = rng.normal(size=(N, K, D))
ADV = rng.normal(size=(N, K))
THETA_OLD = rng.normal(size=D) * 2.0


def mean_kl(theta_a, theta_b):
    pa, pb = probs(PHI, theta_a), probs(PHI, theta_b)
    return np.mean(np.sum(pa * (np.log(pa) - np.log(pb)), axis=1))


def surrogate(theta):
    return np.mean(np.sum(probs(PHI, theta) * ADV, axis=1))


def gradient(theta):
    p = probs(PHI, theta)
    phi_bar = np.einsum("nk,nkd->nd", p, PHI)
    return np.einsum("nk,nk,nkd->d", p, ADV, PHI - phi_bar[:, None, :]) / N


def hessian_vector_fd(theta, v, h=1e-4):
    """Hessian of mean_kl(theta, .) at theta, applied to v, by central differences of the gradient of the KL."""

    def grad_kl(th):
        out = np.zeros(D)
        for i in range(D):
            e = np.eye(D)[i] * h
            out[i] = (mean_kl(theta, th + e) - mean_kl(theta, th - e)) / (2 * h)
        return out

    return (grad_kl(theta + h * v) - grad_kl(theta - h * v)) / (2 * h)


def test_fvp_matches_hessian_of_kl():
    for v in (np.array([1.0, 0.0, 0.0]), np.array([0.3, -0.7, 1.1])):
        want = hessian_vector_fd(THETA_OLD, v)
        got = fisher_vector_product(PHI, THETA_OLD, v)
        assert np.allclose(got, want, atol=1e-5), f"expected {want}, got {got}"


def test_fvp_page_example_matrix_column():
    got = fisher_vector_product(PHI, THETA_OLD, np.array([1.0, 0.0, 0.0]))
    want = np.array([0.1602, 0.0097, -0.0933])
    assert np.allclose(got, want, atol=1e-3), f"expected {want}, got {got}"


def test_fvp_is_linear_and_positive():
    u, v = np.array([1.0, -2.0, 0.5]), np.array([0.2, 0.4, -1.0])
    f = lambda x: fisher_vector_product(PHI, THETA_OLD, x)
    assert np.allclose(f(2 * u + 3 * v), 2 * f(u) + 3 * f(v)), "F v must be linear in v"
    assert u @ f(u) > 0, f"u^T F u must be positive, got {u @ f(u)}"


def test_fvp_is_zero_for_a_constant_shift_of_logits():
    # Moving every logit in a state by the same amount leaves the policy unchanged, so the KL has no curvature there.
    phi = np.ones((2, 3, 1))
    got = fisher_vector_product(phi, np.array([0.7]), np.array([1.0]))
    assert np.allclose(got, 0, atol=1e-12), f"identical features give an unchanged policy, expected 0, got {got}"


def test_real_step_backtracks_once():
    # The full natural-gradient step has KL 0.1316 > 0.1 here, so the line search must shrink beta once.
    delta = 0.1
    fvp = lambda v: fisher_vector_product(PHI, THETA_OLD, v)
    new = trpo_step(THETA_OLD, gradient(THETA_OLD), fvp, delta, surrogate, lambda th: mean_kl(THETA_OLD, th))
    kl_new = mean_kl(THETA_OLD, new)
    gain = surrogate(new) - surrogate(THETA_OLD)
    assert np.isclose(kl_new, 0.0295, atol=1e-3), f"expected KL 0.0295 after one backtrack, got {kl_new}"
    assert np.isclose(gain, 0.1154, atol=1e-3), f"expected surrogate gain 0.1154, got {gain}"


def test_full_step_is_taken_when_the_model_is_right():
    F = np.array([[2.0, 0.0], [0.0, 1.0]])
    g = np.array([1.0, 1.0])
    theta0 = np.zeros(2)
    kl = lambda th: 0.5 * (th - theta0) @ F @ (th - theta0)  # exactly quadratic
    L = lambda th: g @ (th - theta0)  # exactly linear
    new = trpo_step(theta0, g, lambda v: F @ v, 0.1, L, kl)
    s = np.linalg.solve(F, g)
    want = np.sqrt(0.2 / (s @ F @ s)) * s
    assert np.allclose(new, want, atol=1e-6), f"expected the full step {want}, got {new}"


def test_line_search_enforces_the_kl_constraint():
    # The true KL is 6x the quadratic model, so the full step violates the budget; scale 0.25 is the first that fits.
    F = np.eye(2)
    g = np.array([1.0, 0.0])
    kl = lambda th: 3.0 * th @ th
    L = lambda th: g @ th
    new = trpo_step(np.zeros(2), g, lambda v: F @ v, 0.1, L, kl)
    want = 0.25 * np.sqrt(0.2) * g
    assert np.allclose(new, want, atol=1e-6), f"expected scale 0.25 of the full step, {want}, got {new}"
    assert kl(new) <= 0.1, f"KL {kl(new)} exceeds delta = 0.1"


def test_line_search_enforces_improvement():
    # The KL model is exact, but the objective curves downward: x - 10 x^3 falls at the full step 0.4472.
    F = np.eye(2)
    g = np.array([1.0, 0.0])
    kl = lambda th: 0.5 * th @ th
    L = lambda th: th[0] - 10 * th[0] ** 3
    new = trpo_step(np.zeros(2), g, lambda v: F @ v, 0.1, L, kl)
    want = 0.5 * np.sqrt(0.2) * g
    assert np.allclose(new, want, atol=1e-6), f"expected half the full step, {want}, got {new}"
    assert L(new) > L(np.zeros(2)), f"the surrogate must improve, got {L(new)}"


def test_returns_old_parameters_when_nothing_qualifies():
    F = np.eye(2)
    g = np.array([1.0, 0.0])
    theta0 = np.array([0.3, -0.2])
    kl = lambda th: 0.5 * (th - theta0) @ (th - theta0)
    L = lambda th: -np.abs(th - theta0).sum()  # every move makes things worse
    new = trpo_step(theta0, g, lambda v: F @ v, 0.1, L, kl)
    assert np.allclose(new, theta0), f"expected the old parameters {theta0}, got {new}"


def test_shrink_and_backtrack_arguments_are_used():
    F = np.eye(2)
    g = np.array([1.0, 0.0])
    kl = lambda th: 3.0 * th @ th
    L = lambda th: g @ th
    new = trpo_step(np.zeros(2), g, lambda v: F @ v, 0.1, L, kl, shrink=0.1, max_backtracks=3)
    want = 0.1 * np.sqrt(0.2) * g
    assert np.allclose(new, want, atol=1e-6), f"with shrink=0.1 expected scale 0.1, {want}, got {new}"
    none = trpo_step(np.zeros(2), g, lambda v: F @ v, 0.1, L, kl, shrink=0.9, max_backtracks=2)
    assert np.allclose(none, 0), f"two backtracks at 0.9 can't fit the budget, expected theta_old, got {none}"
