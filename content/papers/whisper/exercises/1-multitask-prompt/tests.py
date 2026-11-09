from submission import build_prompt, prompt_loss_mask


def test_transcribe_without_timestamps():
    want = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    got = build_prompt("en", "transcribe", False)
    assert got == want, f"expected {want}, got {got}"


def test_translate_with_timestamps():
    want = ["<|startoftranscript|>", "<|es|>", "<|translate|>"]
    got = build_prompt("es", "translate", True)
    assert got == want, f"expected {want}, got {got}"


def test_language_is_the_source_language():
    got = build_prompt("ko", "translate", False)
    assert got[1] == "<|ko|>", f"the language token names the spoken language; expected '<|ko|>' at index 1, got {got}"
    assert "<|en|>" not in got, f"translate does not add an English token; got {got}"


def test_task_token_order():
    got = build_prompt("de", "transcribe", False)
    assert got.index("<|de|>") < got.index("<|transcribe|>") < got.index("<|notimestamps|>"), f"expected language, then task, then notimestamps; got {got}"


def test_previous_text_comes_first():
    got = build_prompt("en", "transcribe", False, prev_tokens=["Ask", " not"])
    want = ["<|startofprev|>", "Ask", " not", "<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    assert got == want, f"expected {want}, got {got}"


def test_empty_previous_text_adds_no_marker():
    for prev in (None, []):
        got = build_prompt("en", "transcribe", True, prev_tokens=prev)
        assert "<|startofprev|>" not in got, f"prev_tokens={prev!r} must not add <|startofprev|>; got {got}"


def test_input_list_is_not_modified():
    prev = ["a", "b"]
    build_prompt("en", "transcribe", True, prev_tokens=prev)
    assert prev == ["a", "b"], f"prev_tokens must not change; got {prev}"


def test_bad_task():
    try:
        build_prompt("en", "diarize", True)
    except ValueError:
        return
    raise AssertionError("expected ValueError for task 'diarize', but nothing was raised")


def test_mask_hides_only_previous_text():
    toks = ["<|startofprev|>", "Ask", " not", "<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>", " Ask", "<|endoftranscript|>"]
    want = [True, False, False, True, True, True, True, True, True]
    got = prompt_loss_mask(toks)
    assert got == want, f"expected {want}, got {got}"


def test_mask_without_previous_text():
    toks = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>", " Hi", "<|endoftranscript|>"]
    got = prompt_loss_mask(toks)
    assert got == [True] * 6, f"expected six True values, got {got}"
    assert len(got) == len(toks), f"expected one entry per token ({len(toks)}), got {len(got)}"
