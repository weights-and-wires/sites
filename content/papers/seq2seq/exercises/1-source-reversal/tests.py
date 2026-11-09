import numpy as np
from submission import alignment_lags, reverse_source


def test_reverse_source_basic():
    got = reverse_source([1, 2, 3])
    assert list(got) == [3, 2, 1], f"expected [3, 2, 1], got {got}"


def test_reverse_source_does_not_mutate():
    src = [1, 2, 3, 4]
    reverse_source(src)
    assert src == [1, 2, 3, 4], f"the input list must not change; it is now {src}"


def test_lags_forward_are_constant():
    got = alignment_lags(4, False)
    assert np.array_equal(got, [4, 4, 4, 4]), f"expected [4, 4, 4, 4], got {got}"


def test_lags_reversed():
    got = alignment_lags(4, True)
    assert np.array_equal(got, [1, 3, 5, 7]), f"expected [1, 3, 5, 7], got {got}"


def test_matches_brute_force():
    for n in (1, 2, 5, 9):
        src = list(range(n))
        tgt = [100 + j for j in range(n)]
        for rev in (False, True):
            seq = (reverse_source(src) if rev else src) + tgt
            want = [seq.index(100 + j) - seq.index(j) for j in range(n)]
            got = alignment_lags(n, rev)
            assert np.array_equal(got, want), f"n={n} reverse={rev}: expected {want}, got {got}"


def test_average_unchanged_minimum_drops():
    n = 25
    fwd, rev = alignment_lags(n, False), alignment_lags(n, True)
    assert fwd.mean() == rev.mean() == n, f"mean lag must be {n} both ways, got {fwd.mean()} and {rev.mean()}"
    assert fwd.min() == n and rev.min() == 1, f"minimal lag expected {n} and 1, got {fwd.min()} and {rev.min()}"
