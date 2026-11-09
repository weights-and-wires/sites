import numpy as np
from submission import gptq_quantize, make_grid, quant


def layer(seed, d_row=8, d_col=16, m=128, rank=3):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d_row, d_col))
    X = rng.normal(size=(d_col, rank)) @ rng.normal(size=(rank, m)) + 0.3 * rng.normal(size=(d_col, m))
    return W, X


def loss(W, Q, X):
    return float(np.sum(((W - Q) @ X) ** 2))


def rtn(W, bits):
    scale, zero, maxq = make_grid(W, bits)
    return quant(W, scale[:, None], zero[:, None], maxq)


def obq_reference(W, X, bits, percdamp=0.01):
    """Row by row, left to right: Equation 2 for the update and Equation 3 for the inverse."""
    H = 2.0 * X @ X.T
    H = H + percdamp * np.mean(np.diag(H)) * np.eye(H.shape[0])
    scale, zero, maxq = make_grid(W, bits)
    Q = np.zeros_like(W)
    for r in range(W.shape[0]):
        w = W[r].copy()
        Hinv = np.linalg.inv(H)
        alive = list(range(W.shape[1]))
        for q in range(W.shape[1]):
            wq = quant(w[q], scale[r], zero[r], maxq)
            Q[r, q] = wq
            delta = -(w[alive[0]] - wq) / Hinv[0, 0] * Hinv[:, 0]
            w[alive] += delta
            Hinv = (Hinv - np.outer(Hinv[:, 0], Hinv[0, :]) / Hinv[0, 0])[1:, 1:]
            alive = alive[1:]
    return Q


def test_page_example():
    # The page's layer: 16 x 32 weights, 256 samples, four shared input directions.
    rng = np.random.default_rng(0)
    W = rng.normal(size=(16, 32))
    X = rng.normal(size=(32, 4)) @ rng.normal(size=(4, 256)) + 0.3 * rng.normal(size=(32, 256))
    ref = float(np.sum((W @ X) ** 2))
    got = loss(W, gptq_quantize(W, X, 3), X) / ref
    assert abs(got - 0.00518) < 2e-4, f"expected relative layer error about 0.0052, got {got:.5f}"


def test_matches_obq_reference():
    W, X = layer(1)
    got = gptq_quantize(W, X, 3)
    want = obq_reference(W, X, 3)
    assert np.allclose(got, want, atol=1e-9), f"expected the same weights as Equations 2 and 3; max diff {np.abs(got - want).max():.3e}"


def test_first_column_is_plain_rounding():
    W, X = layer(2)
    got = gptq_quantize(W, X, 4)
    want = rtn(W, 4)[:, 0]
    assert np.allclose(got[:, 0], want), f"column 0 has nothing to compensate for yet; expected {want}, got {got[:, 0]}"


def test_output_is_on_the_grid():
    W, X = layer(3)
    Q = gptq_quantize(W, X, 2)
    for r in range(Q.shape[0]):
        assert len(np.unique(np.round(Q[r], 9))) <= 4, f"row {r}: expected at most 4 levels for 2 bits, got {len(np.unique(np.round(Q[r], 9)))}"


def test_beats_round_to_nearest_on_output_error():
    wins = 0
    for seed in range(10):
        W, X = layer(seed + 10)
        wins += loss(W, gptq_quantize(W, X, 3), X) < loss(W, rtn(W, 3), X)
    assert wins == 10, f"expected GPTQ to beat RTN on 10 of 10 layers, got {wins}"


def test_identity_like_inputs_reduce_to_rtn():
    rng = np.random.default_rng(4)
    W = rng.normal(size=(6, 8))
    X = 5.0 * np.concatenate([np.eye(8)] * 4, axis=1)
    # With orthogonal, equal-norm inputs, H is a multiple of the identity, so nothing is compensated.
    got = gptq_quantize(W, X, 3, percdamp=1e-9)
    want = rtn(W, 3)
    assert np.allclose(got, want, atol=1e-6), f"with H proportional to I, expected RTN; max diff {np.abs(got - want).max():.3e}"


def test_inputs_are_not_modified():
    W, X = layer(5)
    W0, X0 = W.copy(), X.copy()
    gptq_quantize(W, X, 3)
    assert np.array_equal(W, W0) and np.array_equal(X, X0), "gptq_quantize must not change W or X"


def test_shape():
    W, X = layer(6, d_row=5, d_col=12)
    assert gptq_quantize(W, X, 3).shape == (5, 12), f"expected shape (5, 12), got {gptq_quantize(W, X, 3).shape}"
