from submission import score_traces, verify_chain

# Figure 1 (right) of the paper: the chain of thought for the cafeteria question.
FIG1 = "The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9."
# Appendix Figure 10, PaLM 62B on the wire question: valid arithmetic, wrong reading of the question.
WIRE_62B = "She cut the wire into 6 inch pieces. This means she got 4 * 6 = 24 pieces. The answer is 24."
# Appendix Figure 10, PaLM 540B on the same question.
WIRE_540B = "The wire was 4 feet long. This means it was 4 * 12 = 48 inches long. It was cut into pieces 6 inches long. This means she obtained 48 / 6 = 8 pieces. The answer is 8."
# Invented: a wrong step that a later step happens to repair.
LUCKY = "Roger has 5 balls. 5 + 6 = 12. 12 - 1 = 11. The answer is 11."


def test_valid_chain():
    got = verify_chain(FIG1)
    assert got == (9.0, 2, []), f"expected (9.0, 2, []), got {got}"


def test_wrong_step_is_reported_with_its_text():
    got = verify_chain(LUCKY)
    assert got == (11.0, 2, ["5 + 6 = 12"]), f"expected (11.0, 2, ['5 + 6 = 12']), got {got}"


def test_arithmetic_check_does_not_check_meaning():
    got = verify_chain(WIRE_62B)
    assert got == (24.0, 1, []), f"the verifier checks arithmetic only; expected (24.0, 1, []), got {got}"


def test_no_equations_and_no_answer():
    got = verify_chain("The coin was flipped by Maybelle. So the answer is no.")
    assert got == (None, 0, []), f"expected (None, 0, []), got {got}"


def test_last_answer_wins():
    got = verify_chain("2 + 2 = 4. The answer is 5. Wait, 2 + 3 = 5. The answer is 5.")
    assert got[0] == 5.0 and got[1] == 2, f"expected answer 5.0 from 2 equations, got {got}"
    got = verify_chain("1 + 1 = 2. The answer is 3. The answer is 2.")
    assert got[0] == 2.0, f"the last 'The answer is' counts; expected 2.0, got {got[0]}"


def test_all_four_operators():
    got = verify_chain("6 * 7 = 42. 42 / 6 = 7. 7 - 10 = -3. -3 + 8 = 5. So the answer is 5.")
    assert got == (5.0, 4, []), f"expected (5.0, 4, []), got {got}"
    got = verify_chain("6 * 7 = 43. 42 / 6 = 8. 7 - 10 = 3. 3 + 8 = 10.")
    assert got[2] == ["6 * 7 = 43", "42 / 6 = 8", "7 - 10 = 3", "3 + 8 = 10"], f"every operator must be checked, got {got[2]}"


def test_decimals_and_division():
    got = verify_chain("0.25 * 40 = 10. 40 + 10 = 50. 10 / 4 = 2.5. So the answer is $50.00.")
    assert got == (50.0, 3, []), f"expected (50.0, 3, []), got {got}"
    got = verify_chain("7 / 2 = 4. 7 / 2 = 3.6. 7 / 0 = 0.")
    assert len(got[2]) == 3, f"7/2 is not 4, not 3.6 to one decimal, and 7/0 is undefined; expected 3 bad steps, got {got[2]}"
    got = verify_chain("10 / 3 = 3.3. 0.1 + 0.2 = 0.3.")
    assert got[2] == [], f"a result rounded to the written decimals is fine; expected no bad steps, got {got[2]}"


def test_sentence_period_is_not_a_decimal_point():
    got = verify_chain("5 + 6 = 11. 11 + 1 = 12. The answer is 12.")
    assert got == (12.0, 2, []), f"expected (12.0, 2, []), got {got}"


def test_score_counts():
    traces = [FIG1, WIRE_62B, WIRE_540B, LUCKY, "The answer is 3.", "2 + 2 = 5. The answer is 5."]
    golds = [9, 8, 8, 11, 3, 4]
    got = score_traces(traces, golds)
    assert abs(got["answer_acc"] - 4 / 6) < 1e-12, f"answer_acc: expected 4 of 6, got {got['answer_acc']}"
    assert abs(got["chain_valid"] - 3 / 6) < 1e-12, f"chain_valid: expected 0.5 (a trace with no equations is not valid), got {got['chain_valid']}"
    assert abs(got["lucky_correct"] - 2 / 4) < 1e-12, f"lucky_correct: expected 2 of the 4 correct answers (a wrong answer does not count), got {got['lucky_correct']}"


def test_score_with_nothing_correct():
    got = score_traces([WIRE_62B], [8])
    assert got == {"answer_acc": 0.0, "chain_valid": 1.0, "lucky_correct": 0.0}, f"expected a 0.0 lucky_correct with no correct answers, got {got}"


def test_score_length_mismatch():
    try:
        score_traces([FIG1], [9, 8])
    except ValueError:
        return
    raise AssertionError("expected ValueError for lists of different lengths")
