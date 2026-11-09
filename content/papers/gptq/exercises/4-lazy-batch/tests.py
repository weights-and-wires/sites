import numpy as np
from submission import gptq_blocked, make_grid, quant, inverse_factor


def layer(seed, d_row=8, d_col=24, m=128, rank=3):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d_row, d_col))
    X = rng.normal(size=(d_col, rank)) @ rng.normal(size=(rank, m)) + 0.3 * rng.normal(size=(d_col, m))
    return W, X


def unblocked(W, X, bits, percdamp=0.01):
    W = W.copy()
    scale, zero, maxq = make_grid(W, bits)
    U = inverse_factor(X, percdamp)
    Q = np.zeros_like(W)
    for j in range(W.shape[1]):
        Q[:, j] = quant(W[:, j], scale, zero, maxq)
        err = (W[:, j] - Q[:, j]) / U[j, j]
        W[:, j:] -= np.outer(err, U[j, j:])
    return Q


def test_matches_unblocked_for_every_block_size():
    W, X = layer(0)
    want = unblocked(W, X, 3)
    for B in (1, 2, 5, 8, 24, 128):
        got = gptq_blocked(W, X, 3, blocksize=B)
        assert np.allclose(got, want, atol=1e-9), f"B={B}: expected the unblocked result; max diff {np.abs(got - want).max():.3e}"


def test_ragged_last_block():
    W, X = layer(1, d_col=23)
    want = unblocked(W, X, 4)
    got = gptq_blocked(W, X, 4, blocksize=8)
    assert np.allclose(got, want, atol=1e-9), f"d_col=23, B=8: expected the unblocked result; max diff {np.abs(got - want).max():.3e}"


def test_output_error_is_better_than_rtn():
    W, X = layer(3)
    scale, zero, maxq = make_grid(W, 3)
    r = quant(W, scale[:, None], zero[:, None], maxq)
    g = gptq_blocked(W, X, 3, blocksize=8)
    lr, lg = np.sum(((W - r) @ X) ** 2), np.sum(((W - g) @ X) ** 2)
    assert lg < lr, f"expected GPTQ loss {lg:.3f} below RTN loss {lr:.3f}"


def test_inputs_are_not_modified():
    W, X = layer(4)
    W0 = W.copy()
    gptq_blocked(W, X, 3, blocksize=8)
    assert np.array_equal(W, W0), "gptq_blocked must not change W"


def test_default_block_size_larger_than_matrix():
    W, X = layer(5, d_col=10)
    got = gptq_blocked(W, X, 3)
    want = unblocked(W, X, 3)
    assert np.allclose(got, want, atol=1e-9), f"default block size 128 with 10 columns: max diff {np.abs(got - want).max():.3e}"
