import numpy as np
from submission import fill_batch, group_advantages, keep_group


def test_advantages_of_a_mixed_group():
    got = group_advantages([1, -1, -1, -1])
    want = np.array([1.7321, -0.5774, -0.5774, -0.5774])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_advantages_use_the_population_std():
    got = group_advantages([1, -1])
    assert np.allclose(got, [1.0, -1.0]), f"expected [1, -1] (population std), got {got}"


def test_uniform_groups_have_zero_advantage_and_no_nan():
    for r in ([1, 1, 1, 1], [-1, -1, -1]):
        got = group_advantages(r)
        assert not np.any(np.isnan(got)), f"rewards {r} gave nan: {got}"
        assert np.all(got == 0), f"expected zeros for {r}, got {got}"


def test_advantages_sum_to_zero():
    got = group_advantages([1, 1, -1, 1, -1, -1, -1, 1])
    assert abs(got.sum()) < 1e-12, f"expected the advantages to sum to 0, got {got.sum()}"


def test_keep_group_filters_both_extremes():
    cases = [([1, 1, 1, 1], False), ([-1, -1, -1, -1], False), ([1, -1, -1, -1], True), ([1, 1, 1, -1], True), ([1, -1], True)]
    for r, want in cases:
        got = keep_group(r)
        assert got == want, f"keep_group({r}) expected {want}, got {got}"


def test_fill_batch_skips_uniform_groups_and_counts_draws():
    stream = iter([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, -1], [1, 1]])
    kept, drawn = fill_batch(stream, 2)
    assert kept == [[1, -1], [-1, 1]], f"expected the first two mixed groups, got {kept}"
    assert drawn == 4, f"expected 4 groups read, got {drawn}"


def test_fill_batch_stops_reading_once_full():
    stream = iter([[1, -1]] * 10)
    kept, drawn = fill_batch(stream, 3)
    assert len(kept) == 3 and drawn == 3, f"expected 3 kept and 3 drawn, got {len(kept)} and {drawn}"
    rest = list(stream)
    assert len(rest) == 7, f"expected 7 groups left unread, got {len(rest)}"


def test_fill_batch_handles_an_exhausted_stream():
    kept, drawn = fill_batch(iter([[1, 1], [1, -1], [-1, -1]]), 5)
    assert kept == [[1, -1]], f"expected only one kept group, got {kept}"
    assert drawn == 3, f"expected 3 groups read, got {drawn}"
