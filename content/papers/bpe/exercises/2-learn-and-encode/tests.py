from submission import encode, learn_bpe

FIG1 = {"low": 5, "lowest": 2, "newer": 6, "wider": 3}


def test_first_merges_on_the_figure_1_dictionary():
    got = learn_bpe(FIG1, 4)
    want = [("e", "r"), ("er", "</w>"), ("l", "o"), ("lo", "w")]
    assert got == want, f"expected {want}, got {got}"


def test_ten_merges_on_the_figure_1_dictionary():
    got = learn_bpe(FIG1, 10)
    want = [("e", "r"), ("er", "</w>"), ("l", "o"), ("lo", "w"), ("e", "w"),
            ("ew", "er</w>"), ("n", "ewer</w>"), ("low", "</w>"), ("d", "er</w>"), ("i", "der</w>")]
    assert got == want, f"expected {want}, got {got}"


def test_counts_use_word_frequency():
    got = learn_bpe({"zy": 9, "ab": 1}, 1)
    assert got == [("y", "</w>")], f"the frequent word 'zy' must win; expected [('y', '</w>')], got {got}"


def test_tie_goes_to_smallest_pair():
    got = learn_bpe({"ab": 1, "cd": 1}, 1)
    assert got == [("a", "b")], f"all pairs tie at 1; expected [('a', 'b')], got {got}"


def test_stops_when_nothing_is_left_to_merge():
    got = learn_bpe({"a": 3}, 5)
    assert got == [("a", "</w>")], f"one word of one symbol allows one merge; expected [('a', '</w>')], got {got}"
    assert learn_bpe({"ab": 1}, 0) == [], f"zero merges expected an empty list, got {learn_bpe({'ab': 1}, 0)}"


def test_encode_seen_and_unseen_words():
    merges = learn_bpe(FIG1, 10)
    assert encode("lower", merges) == ["low", "er</w>"], f"expected ['low', 'er</w>'], got {encode('lower', merges)}"
    assert encode("newer", merges) == ["newer</w>"], f"expected ['newer</w>'], got {encode('newer', merges)}"
    got = encode("lowest", merges)
    assert got == ["low", "e", "s", "t", "</w>"], f"expected ['low', 'e', 's', 't', '</w>'], got {got}"


def test_encode_keeps_unknown_characters():
    merges = learn_bpe(FIG1, 10)
    got = encode("xyz", merges)
    assert got == ["x", "y", "z", "</w>"], f"expected ['x', 'y', 'z', '</w>'], got {got}"
    assert encode("abc", []) == ["a", "b", "c", "</w>"], f"with no merges expected characters plus </w>, got {encode('abc', [])}"


def test_end_of_word_symbol_matters():
    merges = [("e", "r"), ("er", "</w>")]
    assert encode("er", merges) == ["er</w>"], f"expected ['er</w>'], got {encode('er', merges)}"
    got = encode("era", merges)
    assert got == ["er", "a", "</w>"], f"'er' inside a word must not take the word-final symbol; expected ['er', 'a', '</w>'], got {got}"


def test_merge_order_is_priority():
    got = encode("abc", [("a", "b"), ("b", "c")])
    assert got == ["ab", "c", "</w>"], f"('a', 'b') was learned first; expected ['ab', 'c', '</w>'], got {got}"
    got = encode("abc", [("b", "c"), ("a", "b")])
    assert got == ["a", "bc", "</w>"], f"('b', 'c') was learned first; expected ['a', 'bc', '</w>'], got {got}"


def test_chained_and_repeated_merges():
    got = encode("aaaa", [("a", "a")])
    assert got == ["aa", "aa", "</w>"], f"expected ['aa', 'aa', '</w>'], got {got}"
    got = encode("abab", [("a", "b"), ("ab", "ab")])
    assert got == ["abab", "</w>"], f"a merge can use a symbol that an earlier merge made; expected ['abab', '</w>'], got {got}"


def test_symbols_join_back_to_the_word():
    merges = learn_bpe(FIG1, 10)
    for w in ["lower", "wide", "slow", "newest", "dew"]:
        got = "".join(encode(w, merges))
        assert got == w + "</w>", f"symbols of {w!r} must join to {w + '</w>'!r}, got {got!r}"
