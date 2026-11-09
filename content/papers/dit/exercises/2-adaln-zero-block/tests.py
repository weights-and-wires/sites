import numpy as np
from submission import layer_norm, modulate, dit_block

d = 4
rng = np.random.default_rng(0)
X = rng.normal(size=(5, d))
C = rng.normal(size=(d,))
attn = lambda h: 2 * h
mlp = lambda h: h ** 2


def test_modulate_matches_page():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    gamma = np.array([0.5, 0.0, -0.5, 1.0])
    beta = np.array([1.0, 1.0, 0.0, 0.0])
    got = modulate(layer_norm(x), beta, gamma)
    want = np.array([[-1.0125, 0.5528, 0.2236, 2.6833]])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_modulate_zero_is_identity():
    h = layer_norm(X)
    got = modulate(h, np.zeros(d), np.zeros(d))
    assert np.allclose(got, h), "zero shift and zero scale must return h; scale enters as (1 + scale)"


def test_identity_at_zero_init():
    got = dit_block(X, C, np.zeros((d, 6 * d)), np.zeros(6 * d), attn, mlp)
    assert np.allclose(got, X), f"zero-initialized block must be the identity\nmax error {np.abs(got - X).max()}"


def test_identity_for_any_condition_at_zero_init():
    for seed in range(3):
        c = np.random.default_rng(seed).normal(size=d) * 10
        got = dit_block(X, c, np.zeros((d, 6 * d)), np.zeros(6 * d), attn, mlp)
        assert np.allclose(got, X), f"seed {seed}: identity must hold for any c; max error {np.abs(got - X).max()}"


def test_gate_scales_the_residual_branch():
    # Only the attention gate is on (b sets gate_attn = 0.5); shift and scale are 0; MLP gate is 0.
    b = np.zeros(6 * d)
    b[2 * d:3 * d] = 0.5
    got = dit_block(X, C, np.zeros((d, 6 * d)), b, attn, mlp)
    want = X + 0.5 * attn(layer_norm(X))
    assert np.allclose(got, want), f"expected x + 0.5 * attn(LN(x))\nexpected {want[0]}\ngot {got[0]}"


def test_chunk_order_and_both_halves():
    b = np.zeros(6 * d)
    b[0 * d:1 * d] = 0.3   # shift_attn
    b[1 * d:2 * d] = 0.2   # scale_attn
    b[2 * d:3 * d] = 1.0   # gate_attn
    b[3 * d:4 * d] = -0.1  # shift_mlp
    b[4 * d:5 * d] = 0.4   # scale_mlp
    b[5 * d:6 * d] = 0.5   # gate_mlp
    got = dit_block(X, np.zeros(d), np.zeros((d, 6 * d)), b, attn, mlp)
    h1 = layer_norm(X) * 1.2 + 0.3
    y = X + 1.0 * attn(h1)
    h2 = layer_norm(y) * 1.4 - 0.1
    want = y + 0.5 * mlp(h2)
    assert np.allclose(got, want), f"expected\n{want[0]}\ngot\n{got[0]}"


def test_condition_changes_output_once_weights_move():
    W = rng.normal(size=(d, 6 * d)) * 0.3
    a = dit_block(X, C, W, np.zeros(6 * d), attn, mlp)
    b2 = dit_block(X, -C, W, np.zeros(6 * d), attn, mlp)
    assert not np.allclose(a, b2), "different conditioning vectors must give different outputs"


def test_same_parameters_for_every_token():
    # The modulation has shape (d,), so permuting tokens permutes the output (the stand-in sublayers act per token).
    b = np.random.default_rng(3).normal(size=6 * d) * 0.3
    perm = np.array([3, 0, 4, 1, 2])
    a = dit_block(X, C, np.zeros((d, 6 * d)), b, attn, mlp)
    p = dit_block(X[perm], C, np.zeros((d, 6 * d)), b, attn, mlp)
    assert np.allclose(a[perm], p), "one set of (shift, scale, gate) must apply to every token"
