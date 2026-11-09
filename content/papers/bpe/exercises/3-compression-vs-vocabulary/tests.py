from submission import corpus_stats

CORPUS = {"low": 5, "lower": 2}
FIG1 = {"low": 5, "lowest": 2, "newer": 6, "wider": 3}
FIG1_MERGES = [("e", "r"), ("er", "</w>"), ("l", "o"), ("lo", "w"), ("e", "w"),
               ("ew", "er</w>"), ("n", "ewer</w>"), ("low", "</w>"), ("d", "er</w>"), ("i", "der</w>")]


def test_no_merges_is_character_level():
    n, v, r = corpus_stats(CORPUS, [])
    assert n == 32, f"5 * 4 + 2 * 6 symbols expected 32, got {n}"
    assert v == 6, f"5 characters plus </w> expected 6, got {v}"
    assert abs(r - 1.0) < 1e-12, f"expected ratio 1.0, got {r}"


def test_three_merges():
    merges = [("l", "o"), ("lo", "w"), ("low", "</w>")]
    n, v, r = corpus_stats(CORPUS, merges)
    assert n == 13, f"low -> 1 token (x5), lower -> low e r </w> = 4 tokens (x2): expected 13, got {n}"
    assert v == 9, f"6 initial symbols plus 3 merges expected 9, got {v}"
    assert abs(r - 32 / 13) < 1e-9, f"expected ratio {32 / 13:.4f}, got {r}"


def test_tokens_are_weighted_by_count():
    n, _, _ = corpus_stats({"ab": 100, "cd": 1}, [("a", "b")])
    assert n == 100 * 2 + 1 * 3, f"expected 203, got {n}"


def test_figure_1_dictionary_after_ten_merges():
    n, v, r = corpus_stats(FIG1, FIG1_MERGES)
    assert n == 27, f"expected 27 tokens, got {n}"
    assert v == 21, f"10 characters plus </w> plus 10 merges expected 21, got {v}"
    assert abs(r - 88 / 27) < 1e-9, f"expected ratio {88 / 27:.4f}, got {r}"


def test_more_merges_make_a_longer_vocabulary_and_a_shorter_text():
    prev_n, prev_v = None, None
    for k in range(0, len(FIG1_MERGES) + 1):
        n, v, _ = corpus_stats(FIG1, FIG1_MERGES[:k])
        if prev_n is not None:
            assert n <= prev_n, f"token count must not grow with more merges: {prev_n} -> {n} at k={k}"
            assert v == prev_v + 1, f"vocabulary must grow by one per merge: {prev_v} -> {v} at k={k}"
        prev_n, prev_v = n, v
