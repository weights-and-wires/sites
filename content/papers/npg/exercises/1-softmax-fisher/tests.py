import numpy as np
from submission import fisher, fisher_softmax, probs


def make_problem():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(3, 3, 3))
    theta = rng.normal(size=3)
    rho = np.array([0.5, 0.3, 0.2])
    return phi, theta, rho


def test_tabular_matrix_from_the_page():
    p = np.array([0.5, 0.3, 0.2])
    want = np.array([[0.25, -0.15, -0.10], [-0.15, 0.21, -0.06], [-0.10, -0.06, 0.16]])
    got = fisher_softmax(p)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_tabular_matrix_is_singular_along_ones():
    p = np.array([0.1, 0.6, 0.3])
    F = fisher_softmax(p)
    assert np.allclose(F, F.T), f"expected a symmetric matrix, got {F}"
    assert np.allclose(F @ np.ones(3), 0.0, atol=1e-12), f"expected F @ ones = 0, got {F @ np.ones(3)}"
    eig = np.linalg.eigvalsh(F)
    assert eig.min() > -1e-12, f"expected a positive semidefinite matrix, got eigenvalues {eig}"


def test_fisher_matches_the_definition():
    phi, theta, rho = make_problem()
    pi = probs(phi, theta)
    S, K, d = phi.shape
    # Build grad log pi(a; s) by finite differences, then average the outer products.
    want = np.zeros((d, d))
    for s in range(S):
        for a in range(K):
            grad = np.zeros(d)
            for i in range(d):
                e = np.zeros(d)
                e[i] = 1e-6
                lp = np.log(probs(phi, theta + e)[s, a])
                lm = np.log(probs(phi, theta - e)[s, a])
                grad[i] = (lp - lm) / 2e-6
            want += rho[s] * pi[s, a] * np.outer(grad, grad)
    got = fisher(phi, theta, rho)
    assert np.allclose(got, want, atol=1e-6), f"expected\n{want}\ngot\n{got}"


def test_fisher_is_symmetric_positive_definite():
    phi, theta, rho = make_problem()
    F = fisher(phi, theta, rho)
    assert np.allclose(F, F.T), f"expected a symmetric matrix, got {F}"
    eig = np.linalg.eigvalsh(F)
    want = np.array([0.0912, 0.1552, 0.3370])
    assert np.allclose(eig, want, atol=1e-4), f"expected eigenvalues {want} (from the page), got {np.round(eig, 4)}"


def test_state_distribution_weights_each_state():
    phi, theta, rho = make_problem()
    F = fisher(phi, theta, rho)
    F2 = fisher(phi, theta, 2.0 * rho)
    assert np.allclose(F2, 2.0 * F), f"expected doubling rho to double F, got {F2} vs {2.0 * F}"
    only0 = fisher(phi, theta, np.array([1.0, 0.0, 0.0]))
    F0 = fisher(phi[:1], theta, np.array([1.0]))
    assert np.allclose(only0, F0), f"expected rho = [1, 0, 0] to use only state 0, got {only0} vs {F0}"
    uniform = fisher(phi, theta, np.ones(3) / 3)
    assert not np.allclose(uniform, F), "expected a non-uniform rho to change F"


def test_shifting_all_features_in_a_state_changes_nothing():
    phi, theta, rho = make_problem()
    F = fisher(phi, theta, rho)
    shifted = phi.copy()
    shifted[1] += np.array([3.0, -2.0, 5.0])  # same shift for every action in state 1
    F2 = fisher(shifted, theta, rho)
    assert np.allclose(F, F2), f"expected the same F after a common shift (the features must be centered), got\n{F2}\nvs\n{F}"


def test_one_hot_features_give_the_tabular_matrix():
    theta = np.log(np.array([0.5, 0.3, 0.2]))
    phi = np.eye(3)[None, :, :]  # one state, one-hot features
    got = fisher(phi, theta, np.array([1.0]))
    want = fisher_softmax(np.array([0.5, 0.3, 0.2]))
    assert np.allclose(got, want), f"expected {want}, got {got}"
