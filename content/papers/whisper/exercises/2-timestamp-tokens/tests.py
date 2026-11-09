from submission import build_target, timestamp_token


def test_exact_grid_points():
    for s, want in [(0.0, "<|0.00|>"), (0.02, "<|0.02|>"), (2.4, "<|2.40|>"), (30.0, "<|30.00|>")]:
        got = timestamp_token(s)
        assert got == want, f"timestamp_token({s}) expected {want}, got {got}"


def test_rounds_to_nearest_step():
    for s, want in [(0.009, "<|0.00|>"), (0.031, "<|0.04|>"), (0.039, "<|0.04|>"), (1.109, "<|1.10|>"), (29.991, "<|30.00|>")]:
        got = timestamp_token(s)
        assert got == want, f"timestamp_token({s}) expected {want} (nearest 20 ms), got {got}"


def test_two_decimals_without_float_noise():
    got = timestamp_token(0.06)
    assert got == "<|0.06|>", f"expected '<|0.06|>', got {got!r}"
    got = timestamp_token(0.58)
    assert got == "<|0.58|>", f"expected '<|0.58|>', got {got!r}"


def test_twenty_ms_not_ten_ms():
    got = timestamp_token(0.07)
    assert got in ("<|0.06|>", "<|0.08|>"), f"0.07 s is on a 10 ms grid only; expected a 20 ms token, got {got}"


def test_with_timestamps():
    segs = [(0.0, 2.4, [" Ask", " not"]), (2.4, 5.0, [" what"])]
    want = ["<|0.00|>", " Ask", " not", "<|2.40|>", "<|2.40|>", " what", "<|5.00|>", "<|endoftranscript|>"]
    got = build_target(segs, True)
    assert got == want, f"expected {want}, got {got}"


def test_without_timestamps():
    segs = [(0.0, 2.4, [" Ask", " not"]), (2.4, 5.0, [" what"])]
    want = [" Ask", " not", " what", "<|endoftranscript|>"]
    got = build_target(segs, False)
    assert got == want, f"expected {want}, got {got}"


def test_no_speech_is_just_the_end_token():
    for ts in (True, False):
        got = build_target([], ts)
        assert got == ["<|endoftranscript|>"], f"timestamps={ts}: expected only the end token, got {got}"


def test_inputs_not_mutated():
    toks = [" a"]
    segs = [(0.0, 1.0, toks)]
    build_target(segs, True)
    assert toks == [" a"], f"the token list must not change; got {toks}"
