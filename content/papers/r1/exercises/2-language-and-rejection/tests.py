from submission import keep_response, language_reward, rejection_sample

OK = "<think>Add 7 and 8 to get 15, carry the 1.</think><answer>85</answer>"


def test_language_reward_values():
    assert language_reward("add the ones digits") == 1.0, f"expected 1.0, got {language_reward('add the ones digits')}"
    mixed = "add the 我们先算 digits"
    assert abs(language_reward(mixed) - 0.75) < 1e-9, f"expected 0.75, got {language_reward(mixed)}"


def test_language_reward_ignores_tokens_without_letters():
    cot = "7 + 8 = 15 so add 15"
    # Counted words: so, add. Both are ASCII.
    assert language_reward(cot) == 1.0, f"numbers and symbols are not words, expected 1.0, got {language_reward(cot)}"
    assert language_reward("7 + 8") == 0.0, f"no words gives 0.0, got {language_reward('7 + 8')}"
    assert language_reward("") == 0.0, f"empty text gives 0.0, got {language_reward('')}"


def test_language_reward_counts_words_not_characters():
    cot = "我们先算先算先算先算 ok ok"
    want = 2 / 3
    assert abs(language_reward(cot) - want) < 1e-9, f"expected {want:.4f} (2 of 3 words), got {language_reward(cot)}"


def test_one_non_ascii_letter_makes_the_word_non_target():
    cot = "the café is open"
    assert abs(language_reward(cot) - 0.75) < 1e-9, f"expected 0.75, got {language_reward(cot)}"


def test_keep_accepts_a_clean_response():
    assert keep_response(OK, 85) is True, "a correct, well-formed, English response must be kept"


def test_keep_rejects_wrong_or_malformed():
    assert keep_response(OK, 86) is False, "wrong answer must be rejected"
    assert keep_response("The answer is 85", 85) is False, "missing tags must be rejected"
    assert keep_response("<think>x</think><answer>185</answer>", 85) is False, "185 is not 85"


def test_keep_rejects_mixed_language_even_if_mostly_english():
    text = "<think>First add the ones digits and then 进位 to the tens digit.</think><answer>85</answer>"
    assert language_reward("First add the ones digits and then 进位 to the tens digit.") < 1.0
    assert keep_response(text, 85) is False, "any non-target word must reject the response"


def test_keep_rejects_code_blocks():
    text = "<think>Run this.\n```\nprint(47 + 38)\n```\nSo it is 85.</think><answer>85</answer>"
    assert keep_response(text, 85) is False, "a code fence in the thinking must reject the response"


def test_keep_rejects_long_paragraphs_but_not_many_short_ones():
    long_para = " ".join(["step"] * 61)
    assert keep_response(f"<think>{long_para}</think><answer>85</answer>", 85) is False, "61 words in one paragraph exceeds 60"
    exactly = " ".join(["step"] * 60)
    assert keep_response(f"<think>{exactly}</think><answer>85</answer>", 85) is True, "60 words is allowed"
    short_paras = "\n\n".join([" ".join(["step"] * 40)] * 3)
    assert keep_response(f"<think>{short_paras}</think><answer>85</answer>", 85) is True, "three 40-word paragraphs are fine"
    assert keep_response(f"<think>{long_para}</think><answer>85</answer>", 85, max_paragraph_words=100) is True, "the limit is a parameter"


def test_rejection_sample_order_and_cap():
    bad = "<think>x</think><answer>1</answer>"
    good = [f"<think>try {'x' * i}</think><answer>85</answer>" for i in range(1, 5)]
    responses = [bad, good[0], bad, good[1], good[2], good[3]]
    got = rejection_sample(responses, 85, 2)
    assert got == good[:2], f"expected the first two keepers {good[:2]}, got {got}"
    assert rejection_sample(responses, 85, 10) == good, f"expected all four keepers, got {rejection_sample(responses, 85, 10)}"
    assert rejection_sample([bad], 85, 3) == [], "no keepers gives an empty list"
