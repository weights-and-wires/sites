from submission import align_text


def test_page_example():
    words = [(1, ["Hello", ","]), (5, ["I", "'", "m"]), (9, ["Moshi"])]
    want = ["EPAD", "Hello", ",", "EPAD", "I", "'", "m", "EPAD", "Moshi", "PAD", "PAD", "PAD"]
    got = align_text(words, 12)
    assert got == want, f"expected {want}, got {got}"


def test_empty_is_all_pad():
    got = align_text([], 4)
    assert got == ["PAD"] * 4, f"expected four PAD, got {got}"


def test_length_is_T():
    got = align_text([(3, ["a"])], 6)
    assert len(got) == 6, f"expected 6 slots, got {len(got)}: {got}"


def test_epad_comes_right_before_the_word():
    got = align_text([(4, ["x", "y"])], 7)
    want = ["PAD", "PAD", "EPAD", "x", "y", "PAD", "PAD"]
    assert got == want, f"expected {want}, got {got}"


def test_word_at_step_one_shifts_by_one():
    got = align_text([(1, ["a", "b"])], 5)
    want = ["EPAD", "a", "b", "PAD", "PAD"]
    assert got == want, f"expected {want}, got {got}"
    assert got[-1] == "PAD", f"the last slot must stay PAD, got {got[-1]}"


def test_epad_never_overwrites_a_text_token():
    got = align_text([(2, ["a", "b"]), (4, ["c"])], 6)
    want = ["EPAD", "a", "b", "c", "PAD", "PAD"]
    assert got == want, f"expected {want}, got {got}"


def test_epad_goes_into_a_free_slot_after_a_gap():
    got = align_text([(2, ["a"]), (4, ["c"])], 6)
    want = ["EPAD", "a", "EPAD", "c", "PAD", "PAD"]
    assert got == want, f"expected {want}, got {got}"


def test_tokens_past_the_end_are_dropped():
    got = align_text([(4, ["a", "b", "c"])], 5)
    want = ["PAD", "PAD", "EPAD", "a", "b"]
    assert got == want, f"expected {want}, got {got}"
