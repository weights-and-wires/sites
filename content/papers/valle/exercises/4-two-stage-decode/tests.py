import numpy as np
from submission import generate

EOS = -1
PROMPT = np.array([[5, 1, 1], [6, 1, 1]])


def make_ar(script):
    calls = []

    def ar_step(phonemes, tokens):
        calls.append(list(tokens))
        n = len(calls) - 1
        return script[n] if n < len(script) else EOS

    return ar_step, calls


def make_nar(T_expected=None):
    calls = []

    def nar_step(phonemes, prompt_codes, prev, j):
        calls.append((j, prev.shape, prompt_codes.shape))
        # Codebook j copies codebook 1 plus j, so the output is easy to check.
        return prev[:, 0] + j

    return nar_step, calls


def test_output_shape_and_values():
    ar, _ = make_ar([7, 8, 9])
    nar, _ = make_nar()
    out = generate(ar, nar, [1, 2], PROMPT, 100, EOS, K=3)
    want = np.array([[7, 9, 10], [8, 10, 11], [9, 11, 12]])
    assert out.shape == (3, 3), f"expected shape (3, 3), got {out.shape}"
    assert np.array_equal(out, want), f"expected\n{want}\ngot\n{out}"


def test_prompt_is_prefix_but_not_output():
    ar, calls = make_ar([7, 8])
    nar, _ = make_nar()
    out = generate(ar, nar, [1], PROMPT, 100, EOS, K=3)
    assert calls[0] == [5, 6], f"first AR call must see the prompt tokens [5, 6], got {calls[0]}"
    assert calls[1] == [5, 6, 7], f"second AR call must see [5, 6, 7], got {calls[1]}"
    assert out[:, 0].tolist() == [7, 8], f"column 0 must hold only generated tokens [7, 8], got {out[:, 0].tolist()}"


def test_eos_is_not_emitted():
    ar, calls = make_ar([7])
    nar, _ = make_nar()
    out = generate(ar, nar, [1], PROMPT, 100, EOS, K=3)
    assert out.shape[0] == 1, f"expected 1 frame, got {out.shape[0]}"
    assert EOS not in out, f"<EOS> must not appear in the output, got {out.tolist()}"
    assert len(calls) == 2, f"expected 2 AR calls (one token, then EOS), got {len(calls)}"


def test_nar_called_once_per_remaining_codebook_with_earlier_codes_only():
    ar, _ = make_ar([7, 8, 9, 4])
    nar, calls = make_nar()
    generate(ar, nar, [1], PROMPT, 100, EOS, K=8)
    stages = [c[0] for c in calls]
    assert stages == [2, 3, 4, 5, 6, 7, 8], f"expected seven NAR calls for j = 2..8, got {stages}"
    for j, shape, pshape in calls:
        assert shape == (4, j - 1), f"stage {j}: prev_codes must have shape (4, {j - 1}), got {shape}"
        assert pshape == (2, 3), f"the full prompt matrix must be passed, got shape {pshape}"


def test_max_len_caps_generation():
    ar, calls = make_ar([3] * 50)
    nar, _ = make_nar()
    out = generate(ar, nar, [1], PROMPT, 5, EOS, K=3)
    assert out.shape == (5, 3), f"expected 5 frames when EOS never comes, got shape {out.shape}"
    assert len(calls) == 5, f"expected exactly 5 AR calls, got {len(calls)}"


def test_empty_generation():
    ar, _ = make_ar([])
    nar, _ = make_nar()
    out = generate(ar, nar, [1], PROMPT, 10, EOS, K=3)
    assert out.shape == (0, 3), f"expected shape (0, 3) when EOS comes first, got {out.shape}"
