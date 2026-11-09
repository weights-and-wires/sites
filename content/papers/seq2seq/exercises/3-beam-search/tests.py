import numpy as np
from submission import beam_search

A, B, EOS = 0, 1, 2
TABLE = {
    (): [0.6, 0.4, 0.0],
    (A,): [0.30, 0.25, 0.45],
    (B,): [0.05, 0.05, 0.90],
    (A, A): [0.35, 0.25, 0.40],
}
DEFAULT = [0.3, 0.2, 0.5]


def step_fn(prefix):
    p = np.array(TABLE.get(tuple(prefix), DEFAULT), dtype=float)
    with np.errstate(divide="ignore"):
        return np.log(p)


def test_beam_one_is_greedy():
    res = beam_search(step_fn, 1, EOS, 6)
    assert len(res) == 1, f"expected one complete hypothesis, got {len(res)}: {res}"
    toks, lp = res[0]
    assert tuple(toks) == (A, EOS), f"expected (0, 2), got {tuple(toks)}"
    assert np.isclose(lp, np.log(0.27)), f"expected {np.log(0.27):.4f}, got {lp}"


def test_wider_beam_finds_better_hypothesis():
    res = beam_search(step_fn, 2, EOS, 6)
    toks, lp = res[0]
    assert tuple(toks) == (B, EOS), f"expected best (1, 2), got {tuple(toks)}"
    assert np.isclose(lp, np.log(0.36)), f"expected {np.log(0.36):.4f}, got {lp}"
    assert [tuple(t) for t, _ in res] == [(B, EOS), (A, EOS)], f"expected [(1, 2), (0, 2)], got {[tuple(t) for t, _ in res]}"


def test_scores_are_sums_of_step_log_probs():
    res = beam_search(step_fn, 3, EOS, 4)
    for toks, lp in res:
        want = 0.0
        for i, t in enumerate(toks):
            want += step_fn(tuple(toks[:i]))[t]
        assert np.isclose(lp, want), f"{tuple(toks)}: expected {want}, got {lp}"


def test_results_sorted_best_first():
    res = beam_search(step_fn, 3, EOS, 4)
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True), f"expected scores in descending order, got {scores}"
    assert len(res) >= 2, f"expected at least 2 complete hypotheses, got {len(res)}"


def test_complete_hypotheses_end_once_in_eos():
    res = beam_search(step_fn, 3, EOS, 5)
    for toks, _ in res:
        assert toks[-1] == EOS and EOS not in toks[:-1], f"expected exactly one EOS, at the end; got {tuple(toks)}"


def test_max_len_respected():
    res = beam_search(step_fn, 3, EOS, 3)
    assert all(len(t) <= 3 for t, _ in res), f"expected lengths <= 3, got {[len(t) for t, _ in res]}"
    none = beam_search(lambda p: np.log(np.array([0.5, 0.5, 0.0]) + 1e-300), 2, EOS, 4)
    assert none == [] or all(t[-1] == EOS for t, _ in none), f"live hypotheses must be dropped, got {none}"


def test_beam_width_limits_live_hypotheses():
    calls = []

    def counting(prefix):
        calls.append(len(prefix))
        return step_fn(prefix)

    beam_search(counting, 2, EOS, 5)
    for depth in range(5):
        n = calls.count(depth)
        assert n <= 2, f"depth {depth}: expected at most 2 expansions, got {n}"
