import numpy as np

from submission import cache_positions, streaming_scores


def ref_scores(q, K, base=10000.0):
    """Independent reference: a rotary score depends only on the position difference."""
    m, d = K.shape
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    qc = q[0::2] + 1j * q[1::2]
    out = np.empty(m)
    for j in range(m):
        kc = K[j, 0::2] + 1j * K[j, 1::2]
        out[j] = np.real(np.sum(qc * np.conj(kc) * np.exp(1j * ((m - 1) - j) * theta)))
    return out


def test_figure_4_positions():
    got = list(cache_positions([0, 1, 2, 3, 6, 7, 8, 9]))
    assert got == [0, 1, 2, 3, 4, 5, 6, 7], f"expected [0, 1, 2, 3, 4, 5, 6, 7], got {got}"


def test_positions_ignore_original_indices():
    a = list(cache_positions([0, 1, 2, 3, 996, 997, 998, 999]))
    b = list(cache_positions([5, 9, 12, 40, 41, 77, 78, 90]))
    assert a == b == list(range(8)), f"expected 0..7 for both, got {a} and {b}"


def test_scores_match_reference():
    rng = np.random.default_rng(0)
    q, K = rng.normal(size=8), rng.normal(size=(7, 8))
    got = streaming_scores(q, K, [0, 1, 2, 3, 6, 7, 8])
    want = ref_scores(q, K)
    assert np.allclose(got, want), f"expected {np.round(want, 4)}, got {np.round(got, 4)}"


def test_scores_do_not_depend_on_original_indices():
    rng = np.random.default_rng(1)
    q, K = rng.normal(size=8), rng.normal(size=(8, 8))
    a = streaming_scores(q, K, [0, 1, 2, 3, 6, 7, 8, 9])
    b = streaming_scores(q, K, [0, 1, 2, 3, 5000, 5001, 5002, 5003])
    assert np.allclose(a, b), f"scores must use cache positions only; expected {np.round(a, 4)}, got {np.round(b, 4)}"


def test_self_score_is_plain_dot_product():
    rng = np.random.default_rng(2)
    q, k = rng.normal(size=6), rng.normal(size=6)
    K = np.vstack([rng.normal(size=(3, 6)), k])
    got = streaming_scores(q, K, [0, 1, 2, 9])[-1]
    assert np.isclose(got, q @ k), f"distance 0 means no net rotation: expected {q @ k:.4f}, got {got:.4f}"


def test_sink_distance_is_cache_distance():
    q = np.array([1.0, 0.0])
    K = np.tile(np.array([1.0, 0.0]), (8, 1))
    got = streaming_scores(q, K, [0, 1, 2, 3, 6, 7, 8, 9])
    assert np.isclose(got[0], np.cos(7.0)), f"sink score: expected cos(7) = {np.cos(7.0):.4f}, got {got[0]:.4f}"
    assert not np.isclose(got[0], np.cos(9.0)), "distance 9 is the original distance; the cache distance is 7"


def test_shape_and_keys_not_modified():
    rng = np.random.default_rng(3)
    q, K = rng.normal(size=4), rng.normal(size=(5, 4))
    K0 = K.copy()
    got = streaming_scores(q, K, list(range(5)))
    assert got.shape == (5,), f"expected shape (5,), got {got.shape}"
    assert np.array_equal(K, K0), "K holds un-rotated keys and must not be modified"
