from submission import beam_search

# The tree of scripts/beam_toy.py. A prefix is a tuple of step indices.
SCORE = {
    (0,): 0.30, (1,): 0.80, (2,): 0.50, (3,): 0.90,
    (1, 0): 0.40, (1, 1): 0.95,
    (3, 0): 0.60, (3, 1): 0.70,
}


def make_expand(log):
    def expand(prefix, m):
        log.append((prefix, m))
        return [prefix + (i,) for i in range(m)]
    return expand


def test_page_example_final_candidates():
    got = beam_search(make_expand([]), SCORE.__getitem__, 4, 2, 2)
    want = [(1, 1), (3, 1), (3, 0), (1, 0)]
    assert got == want, f"expected {want}, got {got}"


def test_keeps_the_top_prefixes_not_the_bottom():
    log = []
    beam_search(make_expand(log), SCORE.__getitem__, 4, 2, 2)
    kept = [p for p, m in log if p != ()]
    assert kept == [(3,), (1,)], f"expected to expand (3,) then (1,), got {kept}"


def test_expand_call_pattern():
    log = []
    beam_search(make_expand(log), SCORE.__getitem__, 4, 2, 2)
    want = [((), 4), ((3,), 2), ((1,), 2)]
    assert log == want, f"expected expand calls {want}, got {log}"


def test_one_step_returns_sorted_first_steps():
    log = []
    got = beam_search(make_expand(log), SCORE.__getitem__, 4, 2, 1)
    want = [(3,), (1,), (2,), (0,)]
    assert got == want, f"expected {want}, got {got}"
    assert len(log) == 1, f"expected a single expand call, got {len(log)}"


def test_three_rounds_use_n_over_m_kept_prefixes_each_round():
    # Score = number of 1s in the prefix, so ties are common and the stable order matters.
    def score(p):
        return float(sum(p))
    log = []
    got = beam_search(make_expand(log), score, 4, 2, 3)
    assert len(got) == 4, f"expected 4 final prefixes, got {len(got)}: {got}"
    assert len(log) == 1 + 2 * 2, f"expected 1 + (3 - 1) * (4 // 2) = 5 expand calls, got {len(log)}"
    assert all(m == 2 for p, m in log[1:]), f"expected beam_width 2 after the first call, got {log}"
    want = [(3, 1, 1), (3, 1, 0), (3, 0, 1), (3, 0, 0)]
    assert got == want, f"expected {want}, got {got}"


def test_beam_width_one_is_greedy_at_each_round():
    log = []
    # N // M = 1 kept prefix, expanded with 4 children. Children missing from SCORE score 0.
    got = beam_search(make_expand(log), lambda p: SCORE.get(p, 0.0), 4, 4, 2)
    assert [p for p, m in log if p != ()] == [(3,)], f"expected only (3,) to be expanded, got {log}"
    assert len(got) == 4, f"expected 4 final prefixes, got {len(got)}"


def test_ties_keep_the_original_order():
    def score(p):
        return 1.0
    got = beam_search(make_expand([]), score, 4, 2, 2)
    want = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert got == want, f"expected {want} (stable order on ties), got {got}"
