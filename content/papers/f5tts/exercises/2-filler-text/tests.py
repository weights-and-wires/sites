from submission import estimate_total_frames, pad_text


def test_pad_page_example():
    z = pad_text(list("hello"), 12)
    want = ["h", "e", "l", "l", "o"] + ["<F>"] * 7
    assert z == want, f"expected {want}, got {z}"


def test_pad_length_and_order():
    z = pad_text(list("abc"), 6, filler="_")
    assert len(z) == 6, f"expected length 6, got {len(z)}"
    assert z[:3] == ["a", "b", "c"], f"characters must come first, got {z}"
    assert z[3:] == ["_"] * 3, f"expected 3 fillers at the end, got {z}"


def test_pad_exact_fit_has_no_filler():
    z = pad_text(list("abc"), 3)
    assert z == ["a", "b", "c"], f"expected no filler, got {z}"


def test_pad_too_long_raises():
    try:
        pad_text(list("abcd"), 3)
    except ValueError:
        return
    raise AssertionError("expected ValueError when characters exceed the frame count")


def test_pad_does_not_modify_input():
    chars = list("abc")
    pad_text(chars, 8)
    assert chars == ["a", "b", "c"], f"input was modified: {chars}"


def test_duration_clean_case():
    got = estimate_total_frames(320, "x" * 40, "y" * 60)
    assert got == 800, f"expected 800, got {got}"


def test_duration_floors():
    got = estimate_total_frames(100, "x" * 30, "y" * 20)
    assert got == 166, f"expected 100 + floor(66.67) = 166, got {got}"
    got2 = estimate_total_frames(10, "x" * 3, "y" * 1)
    assert got2 == 13, f"expected 10 + floor(3.33) = 13, got {got2}"


def test_duration_page_example_and_type():
    got = estimate_total_frames(320, "Some call me nature, others call me mother nature.", "I am the very model of a modern major general.")
    assert got == 614, f"expected 614, got {got}"
    assert isinstance(got, int), f"expected an int, got {type(got).__name__}"


def test_duration_uses_reference_as_denominator():
    a = estimate_total_frames(200, "x" * 10, "y" * 40)
    assert a == 1000, f"expected 200 + 800 = 1000, got {a}"
    b = estimate_total_frames(200, "x" * 40, "y" * 10)
    assert b == 250, f"expected 200 + 50 = 250, got {b}"
