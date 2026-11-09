from submission import accuracy_reward, extract_answer, format_reward, rule_reward

GOOD = "<think>40 + 30 = 70 and 7 + 8 = 15, so 85.</think><answer>85</answer>"


def test_extract_basic():
    assert extract_answer(GOOD) == "85", f"expected '85', got {extract_answer(GOOD)!r}"
    assert extract_answer("no tags here, 85") is None, f"expected None, got {extract_answer('no tags here, 85')!r}"


def test_extract_takes_the_last_pair_and_strips():
    text = "<answer>13</answer> wait <answer>\n 12 \n</answer>"
    assert extract_answer(text) == "12", f"expected the last pair '12', got {extract_answer(text)!r}"


def test_accuracy_exact_integer():
    assert accuracy_reward(GOOD, 85) == 1.0, f"expected 1.0, got {accuracy_reward(GOOD, 85)}"
    assert accuracy_reward(GOOD, 86) == 0.0, f"expected 0.0, got {accuracy_reward(GOOD, 86)}"


def test_accuracy_is_not_a_substring_match():
    text = "<think>x</think><answer>112</answer>"
    assert accuracy_reward(text, 12) == 0.0, f"112 must not match gold 12, got {accuracy_reward(text, 12)}"
    text = "<think>x</think><answer>The sum is 12</answer>"
    assert accuracy_reward(text, 12) == 0.0, f"prose in the answer tag earns 0, got {accuracy_reward(text, 12)}"


def test_accuracy_reads_only_the_answer_tag():
    text = "<think>the answer is 85</think><answer>7</answer>"
    assert accuracy_reward(text, 85) == 0.0, f"a number in the think block does not count, got {accuracy_reward(text, 85)}"
    assert accuracy_reward("The answer is 85.", 85) == 0.0, "an answer outside the tags earns 0"


def test_accuracy_ignores_leading_zeros_and_handles_negatives():
    assert accuracy_reward("<answer>085</answer>", 85) == 1.0, f"'085' is the integer 85, got {accuracy_reward('<answer>085</answer>', 85)}"
    assert accuracy_reward("<answer>-3</answer>", -3) == 1.0, f"expected 1.0 for -3, got {accuracy_reward('<answer>-3</answer>', -3)}"


def test_format_accepts_whitespace_and_newlines():
    text = "\n<think>step one\nstep two</think>\n\n<answer>85</answer>\n"
    assert format_reward(text) == 1.0, f"expected 1.0, got {format_reward(text)}"


def test_format_rejects_bad_structure():
    bad = {
        "no tags": "85",
        "wrong order": "<answer>85</answer><think>x</think>",
        "no answer tag": "<think>x</think> 85",
        "empty think": "<think>  </think><answer>85</answer>",
        "empty answer": "<think>x</think><answer> </answer>",
        "trailing text": "<think>x</think><answer>85</answer> Hope this helps!",
        "leading text": "Sure. <think>x</think><answer>85</answer>",
        "two think blocks": "<think>x</think><think>y</think><answer>85</answer>",
        "two answer blocks": "<think>x</think><answer>85</answer><answer>85</answer>",
    }
    for name, text in bad.items():
        assert format_reward(text) == 0.0, f"{name}: expected 0.0, got {format_reward(text)} for {text!r}"


def test_rewards_are_independent_and_summed():
    cases = [
        (GOOD, 85, 2.0),
        ("<think>x</think><answer>75</answer>", 85, 1.0),
        ("<answer>85</answer><think>x</think>", 85, 1.0),
        ("The answer is 85.", 85, 0.0),
    ]
    for text, gold, want in cases:
        got = rule_reward(text, gold)
        assert got == want, f"expected {want}, got {got} for {text!r}"
