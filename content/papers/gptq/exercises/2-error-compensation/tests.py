import numpy as np
from submission import compensate

H = np.array([[4.0, 2.0, 1.0, 0.0],
              [2.0, 6.0, 2.0, 1.0],
              [1.0, 2.0, 6.0, 2.0],
              [0.0, 1.0, 2.0, 5.0]])
Hinv = np.linalg.inv(H)
w = np.array([0.37, -0.82, 0.55, 0.14])


def test_page_example():
    got = compensate(w, Hinv, 0, 0.25)
    want = np.array([0.25, -0.7817, 0.5613, 0.1278])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_quantized_weight_lands_on_the_grid_value():
    for q in range(4):
        got = compensate(w, Hinv, q, 0.5)
        assert np.isclose(got[q], 0.5), f"q={q}: expected w[q] = 0.5, got {got[q]}"


def test_optimal_among_all_ways_to_fix_one_weight():
    # Minimize d^T H d with d[q] fixed: the other entries solve a linear system.
    for q in range(4):
        wq_hat = np.round(w[q] * 4) / 4
        free = [i for i in range(4) if i != q]
        e = wq_hat - w[q]
        d = np.zeros(4)
        d[q] = e
        d[free] = -np.linalg.solve(H[np.ix_(free, free)], H[free, q] * e)
        got = compensate(w, Hinv, q, wq_hat)
        assert np.allclose(got, w + d, atol=1e-9), f"q={q}: expected {w + d}, got {got}"


def test_loss_is_below_plain_rounding():
    q, wq_hat = 0, 0.25
    plain = w.copy()
    plain[q] = wq_hat
    got = compensate(w, Hinv, q, wq_hat)
    loss = lambda v: 0.5 * (v - w) @ H @ (v - w)
    assert loss(got) < loss(plain), f"expected compensated loss {loss(got):.5f} below plain rounding {loss(plain):.5f}"
    want = 0.5 * (w[q] - wq_hat) ** 2 / Hinv[q, q]
    assert np.isclose(loss(got), want), f"expected loss {want:.5f} (half of e^2 / Hinv[q, q]), got {loss(got):.5f}"


def test_diagonal_hessian_changes_nothing_else():
    D = np.diag([2.0, 3.0, 4.0, 5.0])
    got = compensate(w, np.linalg.inv(D), 2, 0.5)
    want = w.copy()
    want[2] = 0.5
    assert np.allclose(got, want), f"with a diagonal Hessian the other weights must stay put; expected {want}, got {got}"


def test_input_not_modified():
    w0 = w.copy()
    compensate(w0, Hinv, 1, -0.75)
    assert np.array_equal(w0, w), "compensate must not change w in place"
