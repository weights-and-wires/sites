import numpy as np

from submission import sample_starts, span_mask


def test_hand_example_with_overlap():
    mask = span_mask(12, [1, 4, 6], 3)
    want = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    assert mask.dtype == bool, f"dtype: expected bool, got {mask.dtype}"
    assert np.array_equal(mask, want), f"expected {want.astype(int).tolist()}, got {mask.astype(int).tolist()}"
    assert mask.sum() == 8, f"overlapping spans merge: expected 8 masked steps, got {mask.sum()}"


def test_span_is_cut_off_at_the_end():
    mask = span_mask(10, [8], 5)
    want = np.zeros(10, dtype=bool)
    want[8:] = True
    assert mask.shape == (10,), f"length: expected (10,), got {mask.shape}"
    assert np.array_equal(mask, want), f"expected {want.astype(int).tolist()}, got {mask.astype(int).tolist()}"


def test_span_includes_its_start_and_covers_exactly_M():
    mask = span_mask(20, [5], 10)
    assert mask[5] and mask[14] and not mask[15] and not mask[4], f"span [5, 15): got {mask.astype(int).tolist()}"
    assert mask.sum() == 10, f"one span masks M = 10 steps, got {mask.sum()}"


def test_no_starts_masks_nothing():
    mask = span_mask(7, [], 10)
    assert mask.shape == (7,) and not mask.any(), f"expected 7 False values, got {mask.astype(int).tolist()}"


def test_sample_starts_count_distinct_sorted_in_range():
    rng = np.random.default_rng(0)
    s = sample_starts(781, 0.065, rng)
    assert len(s) == 51, f"round(0.065 * 781) = 51 starts expected, got {len(s)}"
    assert len(set(s.tolist())) == 51, "starts must be distinct (sampled without replacement)"
    assert np.all(np.diff(s) > 0), f"starts must be sorted: got {s[:8].tolist()}..."
    assert s.min() >= 0 and s.max() < 781, f"starts must lie in [0, 781), got min {s.min()} max {s.max()}"


def test_masked_fraction_is_about_49_percent():
    rng = np.random.default_rng(1)
    fracs = [span_mask(781, sample_starts(781, 0.065, rng), 10).mean() for _ in range(300)]
    m = float(np.mean(fracs))
    assert 0.46 < m < 0.52, f"p = 0.065, M = 10 should mask about 0.49 of the steps, got {m:.4f}"
    assert m < 0.065 * 10, f"overlaps must make the fraction less than p * M = 0.65, got {m:.4f}"


def test_sampling_uses_the_generator():
    a = sample_starts(100, 0.1, np.random.default_rng(3))
    b = sample_starts(100, 0.1, np.random.default_rng(3))
    c = sample_starts(100, 0.1, np.random.default_rng(4))
    assert np.array_equal(a, b), "the same seed must give the same starts"
    assert not np.array_equal(a, c), "different seeds should give different starts"
