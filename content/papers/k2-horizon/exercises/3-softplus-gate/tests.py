import numpy as np

from submission import softplus, gated_output

LN2 = np.log(2)


def test_identity_at_zero():
    got = softplus(np.array([0.0]), LN2)
    assert np.allclose(got, 1.0), f"softplus(0; ln 2) must be 1, got {got}"


def test_worked_values():
    got = softplus(np.array([-4.0, -1.0, 1.0, 4.0]), LN2)
    want = np.array([0.0875, 0.5850, 1.5850, 4.0875])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {np.round(got, 4)}"


def test_beta_scales_correctly():
    got = softplus(np.array([1.0]), 2.0)
    want = np.log1p(np.exp(2.0)) / 2.0
    assert np.allclose(got, want), f"expected (1/beta) log(1 + e^(beta x)) = {want:.4f}, got {got}"


def test_no_overflow():
    got = softplus(np.array([1000.0, -1000.0]), LN2)
    assert np.all(np.isfinite(got)), f"overflow: {got}"
    assert np.isclose(got[0], 1000.0, rtol=1e-6), f"large x should give about x, got {got[0]}"
    assert got[1] >= 0, f"softplus is never negative, got {got[1]}"


def test_zero_weights_leave_output_unchanged():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(4, 6))
    X = rng.normal(size=(4, 5))
    got = gated_output(Y, X, np.zeros((5, 6)), np.zeros(6))
    assert np.allclose(got, Y), "with zero gate weights, the softplus gate must return Y unchanged"


def test_gate_can_amplify():
    Y = np.ones((1, 2))
    X = np.ones((1, 1))
    got = gated_output(Y, X, np.array([[3.0, -3.0]]), np.zeros(2))
    assert got[0, 0] > 1.0, f"a positive pre-activation should amplify, got {got[0, 0]:.4f}"
    assert 0 < got[0, 1] < 1.0, f"a negative pre-activation should attenuate, got {got[0, 1]:.4f}"
