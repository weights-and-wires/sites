import numpy as np
from submission import actor_gradient, ascend

k = np.array([1.0, -0.5])
theta = np.array([0.2, 0.4])
S = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def J(th, S=S, k=k):
    return np.mean(-((S @ th - S @ k) ** 2))


def finite_diff(th, S=S, k=k, eps=1e-6):
    return np.array([(J(th + eps * e, S, k) - J(th - eps * e, S, k)) / (2 * eps) for e in np.eye(len(th))])


def test_gradient_on_page_example():
    g = np.asarray(actor_gradient(theta, S, k))
    want = np.array([0.4667, -0.6667])
    assert g.shape == (2,), f"expected shape (2,), got {g.shape}"
    assert np.allclose(g, want, atol=1e-3), f"expected {want}, got {g}"


def test_matches_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(5):
        S2 = rng.normal(size=(6, 3))
        k2 = rng.normal(size=3)
        th = rng.normal(size=3)
        g = np.asarray(actor_gradient(th, S2, k2))
        fd = finite_diff(th, S2, k2)
        assert np.allclose(g, fd, atol=1e-5), f"chain-rule gradient {g} differs from finite differences {fd}"


def test_gradient_is_zero_at_the_optimum():
    g = np.asarray(actor_gradient(k.copy(), S, k))
    assert np.allclose(g, 0.0, atol=1e-12), f"mu(s) = k . s is the best actor, so the gradient must be 0, got {g}"


def test_gradient_is_a_mean_over_states():
    g1 = np.asarray(actor_gradient(theta, S[:1], k))
    g3 = np.asarray(actor_gradient(theta, np.vstack([S[:1]] * 3), k))
    assert np.allclose(g1, g3), f"repeating one state must not change the mean gradient: {g1} vs {g3}"


def test_one_ascent_step_raises_j():
    new = np.asarray(ascend(theta, S, k, 0.1, 1))
    want = theta + 0.1 * finite_diff(theta)
    assert np.allclose(new, want, atol=1e-5), f"expected one step to {want}, got {new}"
    assert J(new) > J(theta), f"J must increase: {J(theta)} -> {J(new)}"


def test_ascent_converges_to_the_best_linear_actor():
    new = np.asarray(ascend(theta, S, k, 0.1, 300))
    assert np.allclose(new, k, atol=1e-3), f"expected theta close to k = {k}, got {new}"


def test_ascend_does_not_modify_its_input():
    th = theta.copy()
    ascend(th, S, k, 0.1, 5)
    assert np.allclose(th, theta), f"the input theta changed: {th}"
