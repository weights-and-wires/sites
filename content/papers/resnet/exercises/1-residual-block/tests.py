import numpy as np
from submission import residual_block

W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
W2 = np.array([[1.0, -1.0], [0.5, 0.5]])
x = np.array([2.0, -1.0])


def test_identity_shortcut_by_hand():
    # relu(W1 x) = [2, 0]; F = W2 [2, 0] = [2, 1]; F + x = [4, 0]; relu -> [4, 0]
    got = residual_block(x, W1, W2)
    want = np.array([4.0, 0.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_zero_weights_pass_the_input_through():
    Z = np.zeros((3, 3))
    xp = np.array([0.5, 2.0, 7.0])
    got = residual_block(xp, Z, Z)
    assert np.allclose(got, xp), f"with F = 0 and x >= 0 the block is the identity; expected {xp}, got {got}"


def test_second_relu_comes_after_the_addition():
    Z = np.zeros((2, 2))
    xn = np.array([-3.0, 1.0])
    got = residual_block(xn, Z, Z)
    want = np.array([0.0, 1.0])
    assert np.allclose(got, want), f"relu applies to F(x) + x, so the negative entry becomes 0; expected {want}, got {got}"
    # relu(F) + x would give [-3, 1] here, and the sum before the relu differs from the sum after it
    W2n = np.array([[-1.0, 0.0], [0.0, 1.0]])
    W1i = np.eye(2)
    xq = np.array([1.0, 1.0])
    got2 = residual_block(xq, W1i, W2n)
    assert np.allclose(got2, [0.0, 2.0]), f"expected [0, 2] (F = [-1, 1], F + x = [0, 2]), got {got2}"


def test_inner_relu_is_applied():
    xx = np.array([1.0, 1.0])
    W1n = -np.eye(2)  # W1 x = [-1, -1]; relu -> 0, so F = 0
    got = residual_block(xx, W1n, np.eye(2))
    assert np.allclose(got, xx), f"relu(W1 x) is 0 here, so F = 0 and y = x = {xx}; got {got}"


def test_projection_shortcut_changes_dimension():
    W1p = np.ones((4, 2))
    W2p = np.ones((3, 4))
    Ws = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    xp = np.array([1.0, 2.0])
    # relu(W1 x) = [3, 3, 3, 3]; F = [12, 12, 12]; Ws x = [1, 2, 3]
    got = residual_block(xp, W1p, W2p, Ws)
    want = np.array([13.0, 14.0, 15.0])
    assert got.shape == (3,), f"expected shape (3,), got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_shortcut_adds_no_parameters_and_is_not_scaled():
    rng = np.random.default_rng(0)
    xr = rng.normal(size=5)
    W1r, W2r = rng.normal(size=(7, 5)), rng.normal(size=(5, 7))
    want = np.maximum(W2r @ np.maximum(W1r @ xr, 0) + xr, 0)
    got = residual_block(xr, W1r, W2r)
    assert np.allclose(got, want), f"expected {want}, got {got}"
