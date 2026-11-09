import numpy as np
from submission import fit_rvq, mse_by_nq, quantize


def two_scale_data(n=400, seed=0):
    """Coarse corners 10 apart plus a fine 2x2 pattern 2 apart, plus tiny noise."""
    rng = np.random.default_rng(seed)
    coarse = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    fine = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    return coarse[rng.integers(4, size=n)] + fine[rng.integers(4, size=n)] + 0.05 * rng.normal(size=(n, 2))


def test_shapes():
    X = two_scale_data()
    books = fit_rvq(X, 3, 4)
    assert len(books) == 3, f"expected 3 codebooks, got {len(books)}"
    assert all(np.asarray(b).shape == (4, 2) for b in books), f"expected shapes (4, 2), got {[np.asarray(b).shape for b in books]}"
    m = mse_by_nq(X, books)
    assert len(m) == 4, f"expected 4 entries (n = 0..3), got {len(m)}"


def test_entry_zero_is_signal_power():
    X = two_scale_data()
    m = mse_by_nq(X, fit_rvq(X, 2, 4))
    want = float(np.mean(X ** 2))
    assert np.isclose(m[0], want), f"entry 0 must be mean(X**2) = {want:.4f}, got {m[0]:.4f}"


def test_matches_explicit_reconstruction():
    X = two_scale_data()
    books = fit_rvq(X, 2, 4)
    x_hat = quantize(books[0], X)
    x_hat = x_hat + quantize(books[1], X - x_hat)
    want = float(np.mean((X - x_hat) ** 2))
    m = mse_by_nq(X, books)
    assert np.isclose(m[2], want), f"entry 2 must equal the error of the two-stage reconstruction {want:.5f}, got {m[2]:.5f}"


def test_error_decreases_with_nq():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 6))
    m = mse_by_nq(X, fit_rvq(X, 5, 8))
    for n in range(1, 6):
        assert m[n] < m[n - 1], f"error must fall as quantizers are added; n={n - 1}: {m[n - 1]:.4f}, n={n}: {m[n]:.4f}"
    assert m[5] < 0.5 * m[1], f"five stages should halve the one-stage error on Gaussian data; got {m[5]:.4f} vs {m[1]:.4f}"


def test_second_stage_fits_the_residual():
    # Stage 1 finds the four coarse corners. Stage 2 must model the fine pattern.
    # A stage 2 fitted on X again would leave the fine structure in the error.
    X = two_scale_data()
    m = mse_by_nq(X, fit_rvq(X, 2, 4))
    assert m[1] > 0.9, f"after stage 1 the fine pattern (squared error about 1 per element) should remain; got {m[1]:.4f}"
    assert m[2] < 0.02, f"after stage 2 only the 0.05 noise should remain (about 0.0025); got {m[2]:.4f}"


def test_partial_use_is_cumulative():
    # Using the first two of three codebooks must not equal using codebook 2 alone.
    X = two_scale_data()
    books = fit_rvq(X, 3, 4)
    m = mse_by_nq(X, books)
    alone = float(np.mean((X - quantize(books[1], X)) ** 2))
    assert m[2] < 0.5 * alone, f"entry 2 must use stages 1 and 2 together ({m[2]:.4f}), not stage 2 alone ({alone:.4f})"
