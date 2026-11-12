from submission import cache_step, evicted_indices, kept_indices


def test_figure_4():
    got = kept_indices(9, 4, 3)
    assert got == [0, 1, 2, 3, 6, 7, 8], f"expected [0, 1, 2, 3, 6, 7, 8], got {got}"
    got = evicted_indices(9, 4, 3)
    assert got == [4, 5], f"expected [4, 5], got {got}"


def test_nothing_evicted_until_full():
    for n in range(0, 13):
        got = kept_indices(n, 4, 8)
        assert got == list(range(n)), f"n={n}: expected every token, got {got}"
        assert evicted_indices(n, 4, 8) == [], f"n={n}: expected no evictions, got {evicted_indices(n, 4, 8)}"


def test_first_eviction_is_token_s():
    got = kept_indices(13, 4, 8)
    want = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
    assert got == want, f"expected {want}, got {got}"
    assert evicted_indices(13, 4, 8) == [4], f"expected [4], got {evicted_indices(13, 4, 8)}"


def test_size_is_constant_and_partition():
    for n in (20, 57, 1000):
        k, e = kept_indices(n, 4, 8), evicted_indices(n, 4, 8)
        assert len(k) == 12, f"n={n}: expected 12 slots, got {len(k)}"
        assert sorted(k + e) == list(range(n)), f"n={n}: kept and evicted must partition range(n)"
        assert k == sorted(k), f"n={n}: kept must be sorted, got {k}"


def test_sinks_never_leave():
    k = kept_indices(10_000, 4, 1020)
    assert k[:4] == [0, 1, 2, 3] and k[-1] == 9999 and len(k) == 1024, f"got head {k[:4]}, tail {k[-1]}, size {len(k)}"


def test_zero_sinks_is_window_attention():
    got = kept_indices(10, 0, 4)
    assert got == [6, 7, 8, 9], f"expected [6, 7, 8, 9], got {got}"
    assert 0 in evicted_indices(10, 0, 4), "window attention evicts token 0"


def test_zero_window_keeps_only_sinks():
    got = kept_indices(10, 2, 0)
    assert got == [0, 1], f"expected [0, 1], got {got}"


def test_cache_step_matches_kept_indices():
    for s, w in ((4, 8), (1, 3), (0, 5), (2, 0), (3, 1)):
        cache = []
        for i in range(40):
            cache = cache_step(cache, i, s, w)
            want = kept_indices(i + 1, s, w)
            assert cache == want, f"s={s}, w={w}, after token {i}: expected {want}, got {cache}"


def test_cache_step_does_not_mutate():
    cache = [0, 1, 2, 3, 6, 7, 8]
    before = list(cache)
    out = cache_step(cache, 9, 4, 3)
    assert cache == before, f"input was modified: expected {before}, got {cache}"
    assert out == [0, 1, 2, 3, 7, 8, 9], f"expected [0, 1, 2, 3, 7, 8, 9], got {out}"
