import numpy as np
from submission import compatible_weights, probs


def make_problem():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(3, 3, 3))
    theta = rng.normal(size=3)
    rho = np.array([0.5, 0.3, 0.2])
    Q = rng.normal(size=(3, 3))
    return phi, theta, rho, Q


def psi_of(phi, theta):
    pi = probs(phi, theta)
    return phi - np.einsum("sk,skd->sd", pi, phi)[:, None, :]


def test_weights_from_the_page():
    phi, theta, rho, Q = make_problem()
    got = compatible_weights(phi, theta, rho, Q)
    want = np.array([0.3890, -0.6130, -0.6926])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {np.round(got, 4)}"


def test_weights_equal_the_natural_gradient():
    phi, theta, rho, Q = make_problem()
    pi = probs(phi, theta)
    psi = psi_of(phi, theta)
    grad = np.einsum("s,sk,skd,sk->d", rho, pi, psi, Q)
    F = np.einsum("s,sk,ska,skb->ab", rho, pi, psi, psi)
    want = np.linalg.solve(F, grad)
    got = compatible_weights(phi, theta, rho, Q)
    assert np.allclose(got, want, atol=1e-8), f"Theorem 1: expected F^-1 grad = {want}, got {got}"


def test_exact_linear_q_is_recovered():
    phi, theta, rho, _ = make_problem()
    w0 = np.array([1.0, -2.0, 0.5])
    Q = psi_of(phi, theta) @ w0
    got = compatible_weights(phi, theta, rho, Q)
    assert np.allclose(got, w0, atol=1e-8), f"expected {w0}, got {got}"


def test_a_per_state_constant_in_q_changes_nothing():
    phi, theta, rho, Q = make_problem()
    w = compatible_weights(phi, theta, rho, Q)
    w2 = compatible_weights(phi, theta, rho, Q + np.array([10.0, -3.0, 7.0])[:, None])
    assert np.allclose(w, w2, atol=1e-8), f"expected the same weights after shifting Q by a constant per state, got {w2} vs {w}"


def test_the_fit_is_weighted_by_rho_and_pi():
    phi, theta, rho, Q = make_problem()
    got = compatible_weights(phi, theta, rho, Q)
    X = psi_of(phi, theta).reshape(9, 3)
    unweighted, *_ = np.linalg.lstsq(X, Q.reshape(9), rcond=None)
    assert not np.allclose(got, unweighted, atol=1e-3), f"expected a weighted fit, but got the unweighted answer {unweighted}"
    pi = probs(phi, theta)
    rho2 = np.array([0.1, 0.1, 0.8])
    W = (rho2[:, None] * pi).reshape(9)
    want = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * Q.reshape(9)))  # normal equations
    got2 = compatible_weights(phi, theta, rho2, Q)
    assert np.allclose(got2, want, atol=1e-8), f"expected the weights rho(s) pi(a; s) = {want}, got {got2}"
