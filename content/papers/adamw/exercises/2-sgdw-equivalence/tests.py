import numpy as np
from submission import l2_factor, sgd_l2_step, sgdw_step

H = np.array([1.0, 4.0, 0.25])
B = np.array([1.0, -2.0, 3.0])


def grad(th):
    return H * (th - B)


def run(step, alpha, lam, beta1, n):
    th, m = np.full(3, 5.0), np.zeros(3)
    for _ in range(n):
        th, m = step(th, grad(th), m, alpha, lam, beta1)
    return th


def test_l2_factor_is_lambda_over_alpha():
    assert np.isclose(l2_factor(0.01, 0.1), 0.1), f"expected 0.1, got {l2_factor(0.01, 0.1)}"
    assert np.isclose(l2_factor(0.01, 0.05), 0.2), f"expected 0.2, got {l2_factor(0.01, 0.05)}"


def test_single_step_values():
    th, g, m = np.array([2.0]), np.array([0.5]), np.array([0.1])
    out, m_new = sgd_l2_step(th, g, m, 0.1, 0.2, 0.9)
    want_m = 0.9 * 0.1 + 0.1 * (0.5 + 0.2 * 2.0)
    assert np.allclose(m_new, want_m) and np.allclose(out, 2.0 - want_m), f"L2: expected m={want_m}, theta={2.0 - want_m}, got m={m_new}, theta={out}"
    out, m_new = sgdw_step(th, g, m, 0.1, 0.2, 0.9)
    want_m = 0.9 * 0.1 + 0.1 * 0.5
    want_t = 2.0 - want_m - 0.2 * 2.0
    assert np.allclose(m_new, want_m) and np.allclose(out, want_t), f"SGDW: expected m={want_m}, theta={want_t}, got m={m_new}, theta={out}"


def test_without_momentum_decay_equals_l2_with_rescaled_factor():
    alpha, lam = 0.1, 0.01
    a = run(sgdw_step, alpha, lam, 0.0, 50)
    b = run(sgd_l2_step, alpha, l2_factor(lam, alpha), 0.0, 50)
    assert np.allclose(a, b, atol=1e-12), f"Proposition 1: expected equal iterates, got {a} and {b}"


def test_unscaled_l2_factor_is_not_equivalent():
    alpha, lam = 0.1, 0.01
    a = run(sgdw_step, alpha, lam, 0.0, 50)
    b = run(sgd_l2_step, alpha, lam, 0.0, 50)
    assert np.abs(a - b).max() > 0.1, f"expected lambda' = lambda to differ from decay lambda, got {a} and {b}"


def test_equivalence_tracks_the_learning_rate():
    # With the same decay, halving alpha must double lambda'.
    for alpha in (0.1, 0.05):
        a = run(sgdw_step, alpha, 0.01, 0.0, 30)
        b = run(sgd_l2_step, alpha, l2_factor(0.01, alpha), 0.0, 30)
        assert np.allclose(a, b, atol=1e-12), f"alpha={alpha}: expected equal iterates, got {a} and {b}"


def test_with_momentum_the_two_variants_differ():
    alpha, lam = 0.1, 0.01
    a = run(sgdw_step, alpha, lam, 0.9, 50)
    b = run(sgd_l2_step, alpha, l2_factor(lam, alpha), 0.9, 50)
    assert np.abs(a - b).max() > 0.1, f"expected momentum to break the equivalence (Algorithm 1), got {a} and {b}"


def test_sgdw_decay_stays_out_of_the_momentum():
    _, m = sgdw_step(np.array([10.0]), np.array([0.0]), np.array([0.0]), 0.1, 0.5, 0.9)
    assert np.allclose(m, 0.0), f"expected m = 0 for zero gradient, got {m}"


def test_inputs_are_not_modified():
    for fn in (sgd_l2_step, sgdw_step):
        th, g, m = np.ones(3), np.array([1.0, 2.0, 3.0]), np.full(3, 0.5)
        copies = [a.copy() for a in (th, g, m)]
        fn(th, g, m, 0.1, 0.1, 0.9)
        for name, before, after in zip("theta grad m".split(), copies, (th, g, m)):
            assert np.array_equal(before, after), f"{fn.__name__} modified {name}: before {before}, after {after}"
