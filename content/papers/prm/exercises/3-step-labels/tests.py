from submission import outcome_labels, process_labels, synthetic_step_ok


def test_process_labels_stop_at_the_first_error():
    got = process_labels([True, True, False, True, False])
    assert got == [1, 1, 0], f"expected [1, 1, 0], got {got}"


def test_process_labels_keep_every_step_when_all_are_correct():
    got = process_labels([True, True, True, True])
    assert got == [1, 1, 1, 1], f"expected [1, 1, 1, 1], got {got}"


def test_process_labels_error_at_the_first_and_last_step():
    got = process_labels([False, True, True])
    assert got == [0], f"expected [0], got {got}"
    got = process_labels([True, True, False])
    assert got == [1, 1, 0], f"expected [1, 1, 0], got {got}"


def test_process_labels_are_integers_and_empty_input_is_empty():
    got = process_labels([True, False])
    assert got == [1, 0] and all(type(x) is int for x in got), f"expected the ints [1, 0], got {got!r}"
    assert process_labels([]) == [], f"expected [], got {process_labels([])}"


def test_outcome_labels_repeat_the_final_label():
    got = outcome_labels(5, False)
    assert got == [0, 0, 0, 0, 0], f"expected five zeros, got {got}"
    got = outcome_labels(3, True)
    assert got == [1, 1, 1], f"expected three ones, got {got}"


def test_outcome_label_ignores_where_the_error_is():
    steps = [True, False, True]  # the page's right-answer, flawed-step solution
    got = outcome_labels(len(steps), True)
    assert got == [1, 1, 1], f"expected [1, 1, 1] (a false positive), got {got}"
    assert process_labels(steps) == [1, 0], f"expected process labels [1, 0], got {process_labels(steps)}"


def test_outcome_labels_do_not_share_one_list():
    a = outcome_labels(2, True)
    b = outcome_labels(2, True)
    a[0] = 9
    assert b == [1, 1], f"expected independent lists, second list is {b}"


def test_synthetic_step_ok_page_example():
    got = synthetic_step_ok([0.05, 0.10, 0.20, 0.25, 0.01])
    assert got == [True, True, True, False, True], f"expected [T, T, T, F, T], got {got}"


def test_synthetic_threshold_is_strict_and_adjustable():
    assert synthetic_step_ok([0.2]) == [True], f"p_neg = 0.2 is not above 0.2, got {synthetic_step_ok([0.2])}"
    assert synthetic_step_ok([0.2000001]) == [False], f"expected [False], got {synthetic_step_ok([0.2000001])}"
    got = synthetic_step_ok([0.05, 0.30], threshold=0.5)
    assert got == [True, True], f"expected [True, True] at threshold 0.5, got {got}"
    got = synthetic_step_ok([0.05, 0.30], threshold=0.1)
    assert got == [True, False], f"expected [True, False] at threshold 0.1, got {got}"


def test_synthetic_pipeline_matches_the_page():
    ok = synthetic_step_ok([0.05, 0.10, 0.20, 0.25, 0.01])
    assert process_labels(ok) == [1, 1, 1, 0], f"expected [1, 1, 1, 0], got {process_labels(ok)}"
    assert outcome_labels(len(ok), all(ok)) == [0] * 5, f"expected five zeros, got {outcome_labels(len(ok), all(ok))}"
