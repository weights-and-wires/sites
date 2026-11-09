import numpy as np
from submission import conjugate_gradient, max_step_length

A2 = np.array([[4.0, 1.0], [1.0, 3.0]])
B2 = np.array([1.0, 2.0])


def random_spd(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M @ M.T + 0.5 * np.eye(n), rng.normal(size=n)


def test_one_iteration_page_example():
    got = conjugate_gradient(lambda v: A2 @ v, B2, iters=1)
    assert np.allclose(got, [0.25, 0.5]), f"expected [0.25, 0.5] after one iteration, got {got}"


def test_two_iterations_solve_a_2x2_system():
    got = conjugate_gradient(lambda v: A2 @ v, B2, iters=2)
    want = np.linalg.solve(A2, B2)
    assert np.allclose(got, want, atol=1e-9), f"expected {want} after 2 iterations, got {got}"


def test_n_iterations_solve_an_n_system():
    for n, seed in [(5, 0), (8, 1)]:
        A, b = random_spd(n, seed)
        got = conjugate_gradient(lambda v: A @ v, b, iters=n, tol=1e-20)
        want = np.linalg.solve(A, b)
        assert np.allclose(got, want, atol=1e-6), f"n={n}: expected {want}, got {got}"


def test_only_matrix_vector_products_are_used():
    A, b = random_spd(6, 2)
    calls = []

    def fvp(v):
        calls.append(1)
        return A @ v

    conjugate_gradient(fvp, b, iters=3)
    assert len(calls) == 3, f"3 iterations should call fvp exactly 3 times, got {len(calls)}"


def test_stops_early_when_converged():
    calls = []

    def fvp(v):
        calls.append(1)
        return 2.0 * v

    got = conjugate_gradient(fvp, np.array([2.0, 4.0, 6.0]), iters=10)
    assert np.allclose(got, [1.0, 2.0, 3.0]), f"expected [1, 2, 3], got {got}"
    assert len(calls) == 1, f"A = 2I converges in one iteration, but fvp was called {len(calls)} times"


def test_zero_right_hand_side():
    got = conjugate_gradient(lambda v: A2 @ v, np.zeros(2), iters=5)
    assert np.allclose(got, 0) and np.all(np.isfinite(got)), f"expected zeros, got {got}"


def test_does_not_modify_b():
    b = B2.copy()
    conjugate_gradient(lambda v: A2 @ v, b, iters=2)
    assert np.array_equal(b, B2), f"b was modified: expected {B2}, got {b}"


def test_step_length_page_example():
    s = np.linalg.solve(A2, B2)
    beta = max_step_length(s, A2 @ s, 0.01)
    assert np.isclose(beta, 0.121106, atol=1e-5), f"expected 0.121106, got {beta}"


def test_step_length_hits_the_kl_budget():
    A, b = random_spd(4, 3)
    s = np.linalg.solve(A, b)
    for delta in (0.01, 0.5):
        beta = max_step_length(s, A @ s, delta)
        quad_kl = 0.5 * (beta * s) @ A @ (beta * s)
        assert np.isclose(quad_kl, delta), f"expected quadratic KL {delta}, got {quad_kl}"
