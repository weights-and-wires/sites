import numpy as np
from submission import decoupled_precond_step, equivalent_l2, l2_precond_step

S = np.array([0.1, 1.0, 10.0])
B = np.array([1.0, 1.0, 1.0])


def grad(th):
    return th - B


def run(step, arg, steps=200, alpha=0.01):
    th = np.full(3, 2.0)
    for _ in range(steps):
        th = step(th, grad(th), S, alpha, arg)
    return th


def test_single_step_values():
    th, g = np.array([2.0, 2.0]), np.array([1.0, 1.0])
    s = np.array([0.5, 2.0])
    got = l2_precond_step(th, g, s, 0.1, 0.5)
    want = th - 0.1 * (g + 0.5 * th) / s
    assert np.allclose(got, want), f"L2: expected {want}, got {got}"
    got = decoupled_precond_step(th, g, s, 0.1, 0.05)
    want = 0.95 * th - 0.1 * g / s
    assert np.allclose(got, want), f"decoupled: expected {want}, got {got}"


def test_equivalent_l2_values():
    got = equivalent_l2(0.01, 0.01, S)
    assert np.allclose(got, S), f"expected lam / alpha * s = {S}, got {got}"
    got = equivalent_l2(0.02, 0.1, np.array([1.0, 3.0]))
    assert np.allclose(got, [0.2, 0.6]), f"expected [0.2, 0.6], got {got}"


def test_proposition_3_iterates_match():
    a = run(decoupled_precond_step, 0.01)
    b = run(l2_precond_step, equivalent_l2(0.01, 0.01, S))
    assert np.allclose(a, b, atol=1e-12), f"Proposition 3: expected equal iterates {a}, got {b}"


def test_uniform_l2_factor_is_not_equivalent_when_scales_differ():
    a = run(decoupled_precond_step, 0.01)
    b = run(l2_precond_step, np.full(3, 0.01 / 0.01))
    assert np.abs(a - b).max() > 0.5, f"expected one lambda' = lambda / alpha to differ for unequal s, got {a} and {b}"


def test_equal_scales_reduce_to_proposition_1():
    s = np.full(3, 1.0)
    lam_l2 = equivalent_l2(0.01, 0.1, s)
    assert np.allclose(lam_l2, 0.1), f"expected lambda / alpha = 0.1 for s = 1, got {lam_l2}"


def test_decay_rate_per_step_differs_across_scales():
    # L2 with a uniform lambda' shrinks theta by alpha * lambda' / s per step; decoupled shrinks by lam.
    th, zero = np.ones(3), np.zeros(3)
    l2 = 1 - l2_precond_step(th, zero, S, 0.01, 1.0)
    dec = 1 - decoupled_precond_step(th, zero, S, 0.01, 0.01)
    assert np.allclose(l2, [0.1, 0.01, 0.001]), f"expected L2 shrinkage [0.1, 0.01, 0.001], got {l2}"
    assert np.allclose(dec, 0.01), f"expected decoupled shrinkage 0.01 for every coordinate, got {dec}"


def test_inputs_are_not_modified():
    th, g, s = np.ones(3), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])
    copies = [a.copy() for a in (th, g, s)]
    l2_precond_step(th, g, s, 0.1, 0.1)
    decoupled_precond_step(th, g, s, 0.1, 0.1)
    for name, before, after in zip("theta grad s".split(), copies, (th, g, s)):
        assert np.array_equal(before, after), f"{name} was modified: before {before}, after {after}"
