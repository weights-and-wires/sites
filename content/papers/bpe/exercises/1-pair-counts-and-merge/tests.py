from submission import get_stats, merge_vocab

LISTING = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}


def test_counts_are_weighted_by_frequency():
    s = get_stats(LISTING)
    assert s[("e", "s")] == 9, f"('e', 's') occurs in newest (6) and widest (3): expected 9, got {s.get(('e', 's'))}"
    assert s[("l", "o")] == 7, f"('l', 'o') occurs in low (5) and lower (2): expected 7, got {s.get(('l', 'o'))}"
    assert s[("w", "e")] == 8, f"('w', 'e') occurs in lower (2) and newest (6): expected 8, got {s.get(('w', 'e'))}"


def test_end_of_word_symbol_is_part_of_a_pair():
    s = get_stats(LISTING)
    assert s[("t", "</w>")] == 9, f"expected 9 for ('t', '</w>'), got {s.get(('t', '</w>'))}"
    assert s[("w", "</w>")] == 5, f"expected 5 for ('w', '</w>'), got {s.get(('w', '</w>'))}"


def test_no_pairs_across_words():
    s = get_stats({"a b </w>": 1, "c d </w>": 1})
    assert ("</w>", "c") not in s, f"pairs must not cross word boundaries, got {dict(s)}"
    assert len(s) == 4, f"expected 4 distinct pairs, got {len(s)}: {dict(s)}"


def test_repeated_symbol_counts_each_position():
    s = get_stats({"a a a </w>": 2})
    assert s[("a", "a")] == 4, f"'a a a' has two ('a', 'a') pairs; with frequency 2 expected 4, got {s.get(('a', 'a'))}"


def test_single_merge():
    got = merge_vocab(("e", "s"), LISTING)
    want = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w es t </w>": 6, "w i d es t </w>": 3}
    assert got == want, f"expected {want}, got {got}"


def test_merge_matches_whole_symbols_only():
    v = {"a b c </w>": 1, "ab c </w>": 4}
    got = merge_vocab(("b", "c"), v)
    want = {"a bc </w>": 1, "ab c </w>": 4}
    assert got == want, f"the pair ('b', 'c') must not match inside the symbol 'ab'; expected {want}, got {got}"


def test_merge_handles_end_of_word_marker():
    got = merge_vocab(("r", "</w>"), {"l o w e r </w>": 2, "e r r </w>": 1})
    want = {"l o w e r</w>": 2, "e r r</w>": 1}
    assert got == want, f"expected {want}, got {got}"


def test_merge_repeats_and_overlaps():
    got = merge_vocab(("a", "b"), {"a b a b </w>": 3})
    assert got == {"ab ab </w>": 3}, f"expected {{'ab ab </w>': 3}}, got {got}"
    got = merge_vocab(("a", "a"), {"a a a </w>": 1})
    assert got == {"aa a </w>": 1}, f"merging left to right, expected {{'aa a </w>': 1}}, got {got}"


def test_input_not_modified():
    v = dict(LISTING)
    merge_vocab(("l", "o"), v)
    assert v == LISTING, f"merge_vocab must return a new dict; the input changed to {v}"
