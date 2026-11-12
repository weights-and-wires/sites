import numpy as np
from submission import topk_mask


def causal_scores(L, seed):
    rng = np.random.default_rng(seed)
    I = rng.normal(size=(L, L))
    return np.where(np.tril(np.ones((L, L), dtype=bool)), I, -np.inf)


def test_row_counts_handle_short_prefixes():
    I = causal_scores(8, 0)
    got = topk_mask(I, 3).sum(axis=1)
    want = np.minimum(3, np.arange(8) + 1)
    assert np.array_equal(got, want), f"expected counts {want}, got {got}"


def test_never_selects_the_future():
    I = causal_scores(8, 1)
    m = topk_mask(I, 4)
    assert not m[np.triu_indices(8, k=1)].any(), "selected a key with s > t"


def test_selects_the_largest_scores():
    I = np.array(
        [
            [1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.5, 0.25, -np.inf, -np.inf, -np.inf, -np.inf],
            [1.0, 0.75, 0.5, -np.inf, -np.inf, -np.inf],
            [1.0, 0.5, 1.5, 0.0, -np.inf, -np.inf],
            [0.5, 1.25, 0.75, 2.0, 0.25, -np.inf],
            [0.25, 0.125, 0.5, 0.0, 1.0, 0.75],
        ]
    )
    got = topk_mask(I, 3)
    want_sets = [[0], [0, 1], [0, 1, 2], [0, 1, 2], [1, 2, 3], [2, 4, 5]]
    for t, want in enumerate(want_sets):
        have = np.flatnonzero(got[t]).tolist()
        assert have == want, f"row {t}: expected {want}, got {have}"


def test_k_one_picks_the_argmax():
    I = causal_scores(9, 2)
    m = topk_mask(I, 1)
    want = np.argmax(I, axis=1)
    got = m.argmax(axis=1)
    assert m.sum(axis=1).tolist() == [1] * 9, f"expected one key per row, got {m.sum(axis=1)}"
    assert np.array_equal(got, want), f"expected argmax {want}, got {got}"


def test_large_k_keeps_every_visible_key():
    I = causal_scores(6, 3)
    got = topk_mask(I, 100)
    want = np.tril(np.ones((6, 6), dtype=bool))
    assert np.array_equal(got, want), f"expected the causal triangle, got {got.astype(int)}"


def test_negative_scores_still_rank():
    # All visible scores are negative. A rule that thresholds at zero would select nothing.
    I = -np.abs(causal_scores(6, 4)) - 1.0
    I = np.where(np.tril(np.ones((6, 6), dtype=bool)), I, -np.inf)
    got = topk_mask(I, 2).sum(axis=1)
    want = np.minimum(2, np.arange(6) + 1)
    assert np.array_equal(got, want), f"expected counts {want}, got {got}"
