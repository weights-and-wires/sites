import zlib

from submission import compression_ratio, decode_with_fallback, needs_fallback, temperature_schedule

GOOD = "The quick brown fox jumps over the lazy dog while the band plays on."
LOOP = "Thank you. " * 20


def test_ratio_matches_zlib_on_bytes():
    for text in (GOOD, LOOP, "언덕 위에 올라 내려다보면"):
        b = text.encode("utf-8")
        want = len(b) / len(zlib.compress(b))
        got = compression_ratio(text)
        assert abs(got - want) < 1e-9, f"expected {want} for {text!r}, got {got}"


def test_ratio_direction():
    assert compression_ratio(LOOP) > 5, f"a looping transcript compresses well, so its ratio is high; got {compression_ratio(LOOP)}"
    assert compression_ratio(GOOD) < 1.5, f"ordinary text barely compresses; got {compression_ratio(GOOD)}"


def test_ratio_counts_bytes_not_characters():
    text = "언덕 위에 올라 내려다보면 너무나 넓고 넓은 들판"
    want = len(text.encode("utf-8")) / len(zlib.compress(text.encode("utf-8")))
    got = compression_ratio(text)
    assert abs(got - want) < 1e-9, f"Korean text is 3 bytes per syllable; expected {want:.4f}, got {got:.4f}"


def test_repetition_triggers_fallback():
    assert needs_fallback(LOOP, -0.2), f"ratio {compression_ratio(LOOP):.2f} > 2.4 must fall back"


def test_low_probability_triggers_fallback():
    assert needs_fallback(GOOD, -1.5), "avg_logprob -1.5 < -1.0 must fall back"


def test_good_decode_passes():
    assert not needs_fallback(GOOD, -0.3), "a plausible, non-repetitive decode must not fall back"


def test_either_condition_is_enough():
    assert needs_fallback(LOOP, -0.1) and needs_fallback(GOOD, -2.0), "the two tests are joined with 'or', not 'and'"


def test_thresholds_are_strict():
    r = compression_ratio(GOOD)
    assert not needs_fallback(GOOD, -1.0, ratio_threshold=r), f"a ratio equal to the threshold ({r}) passes; '>' is strict"
    assert not needs_fallback(GOOD, -1.0), "avg_logprob equal to -1.0 passes; '<' is strict"
    assert needs_fallback(GOOD, -1.0001), "avg_logprob just below -1.0 must fall back"


def test_custom_thresholds_are_used():
    assert needs_fallback(GOOD, -0.5, ratio_threshold=0.5), "a ratio of about 1 is above a threshold of 0.5"
    assert needs_fallback(GOOD, -0.5, logprob_threshold=-0.2), "-0.5 is below a threshold of -0.2"


def test_schedule_default():
    want = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    got = temperature_schedule()
    assert got == want, f"expected {want}, got {got}"


def test_schedule_other_step():
    got = temperature_schedule(0.0, 0.5, 1.0)
    assert got == [0.0, 0.5, 1.0], f"expected [0.0, 0.5, 1.0], got {got}"


def test_first_good_decode_wins():
    calls = []

    def decode(t):
        calls.append(t)
        return (LOOP, -0.2) if t < 0.4 else (GOOD, -0.5)

    text, lp, t = decode_with_fallback(decode, temperature_schedule())
    assert (text, lp, t) == (GOOD, -0.5, 0.4), f"expected (GOOD, -0.5, 0.4), got {(text, lp, t)}"
    assert calls == [0.0, 0.2, 0.4], f"must stop at the first good decode; calls were {calls}"


def test_no_fallback_when_first_is_good():
    calls = []

    def decode(t):
        calls.append(t)
        return GOOD, -0.2

    _, _, t = decode_with_fallback(decode, temperature_schedule())
    assert t == 0.0 and calls == [0.0], f"expected a single call at 0.0, got calls {calls}"


def test_all_fail_returns_last():
    calls = []

    def decode(t):
        calls.append(t)
        return LOOP, -3.0 + t

    text, lp, t = decode_with_fallback(decode, temperature_schedule())
    assert calls == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], f"expected all six temperatures, got {calls}"
    assert t == 1.0 and abs(lp - -2.0) < 1e-9, f"expected the result at temperature 1.0 with logprob -2.0, got t={t}, logprob={lp}"
