from submission import pretokenize, to_bytes


def check(text, want):
    got = pretokenize(text)
    assert got == want, f"pretokenize({text!r}): expected {want}, got {got}"


def test_dog_variants_share_one_word_piece():
    check("dog dog. dog! dog?", ["dog", " dog", ".", " dog", "!", " dog", "?"])


def test_space_joins_the_next_word():
    check("I have 42 dogs", ["I", " have", " 42", " dogs"])


def test_categories_do_not_merge():
    check("abc123!?x", ["abc", "123", "!?", "x"])


def test_punctuation_runs_stay_together():
    check("wait...what", ["wait", "...", "what"])


def test_apostrophe_splits_in_this_simplified_rule():
    check("don't", ["don", "'", "t"])


def test_only_the_last_space_joins():
    check("a  b", ["a", " ", " b"])
    check("a   b", ["a", "  ", " b"])


def test_newline_does_not_join():
    check("hi\n there", ["hi", "\n", " there"])
    check("hi\nthere", ["hi", "\n", "there"])


def test_trailing_and_leading_space():
    check("x ", ["x", " "])
    check(" x", [" x"])


def test_unicode_letters_stay_together():
    check("café au", ["café", " au"])


def test_pieces_join_back_to_text():
    for text in ["", "a", "  ", "Hello,   world!\n\n  42 times.", "世界 2019!"]:
        got = pretokenize(text)
        assert "".join(got) == text, f"pieces must join back to {text!r}, got {got}"
        assert all(got), f"no piece may be empty, got {got}"


def test_to_bytes():
    got = to_bytes("café")
    assert got == [99, 97, 102, 195, 169], f"expected [99, 97, 102, 195, 169], got {got}"
    assert len(to_bytes("\U0001F600")) == 4, "an emoji is 4 UTF-8 bytes"
    assert to_bytes("") == [], "the empty string has no bytes"
