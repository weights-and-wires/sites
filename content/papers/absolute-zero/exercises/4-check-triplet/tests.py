from submission import check_answer

P = "def f(x):\n    return sum(i for i in range(x) if i % 2 == 0)"


def test_deduction():
    assert check_answer("deduction", P, 5, 6, 6) is True, "deduction: 6 equals o, expected True"
    assert check_answer("deduction", P, 5, 6, 5) is False, "deduction: 5 differs from o, expected False"


def test_abduction_accepts_other_preimages():
    assert check_answer("abduction", P, None, 6, 5) is True, "abduction: f(5) = 6, expected True"
    got = check_answer("abduction", P, None, 6, 6)
    assert got is True, f"abduction: f(6) = 6 too, so a different input is correct; expected True, got {got}"


def test_abduction_rejects_wrong_input():
    assert check_answer("abduction", P, None, 6, 3) is False, "abduction: f(3) = 2, expected False"


def test_abduction_does_not_compare_input_to_original():
    # The original input is 5. An answer equal to the output 6 must not be judged by equality with 5.
    got = check_answer("abduction", P, 5, 6, 6)
    assert got is True, f"abduction is checked by running p, not by comparing with i: expected True, got {got}"


def test_abduction_error_is_incorrect():
    got = check_answer("abduction", P, None, 6, "a")
    assert got is False, f"an input that makes p raise is incorrect: expected False, got {got}"


def test_induction_accepts_equivalent_program():
    pairs = [(1, 0), (2, 0), (3, 2), (4, 2), (8, 12), (9, 20)]
    same = "def f(x):\n    return sum(range(0, x, 2))"
    got = check_answer("induction", None, pairs, None, same)
    assert got is True, f"an equivalent program must pass all pairs: expected True, got {got}"


def test_induction_checks_every_pair():
    pairs = [(1, 0), (2, 0), (3, 2), (4, 2), (8, 12), (9, 20)]
    wrong_late = "def f(x):\n    return sum(range(0, x, 2)) if x < 8 else 0"
    got = check_answer("induction", None, pairs, None, wrong_late)
    assert got is False, f"a program wrong on the last pairs must fail: expected False, got {got}"
    bad = "def f(x):\n    return x // 2"
    assert check_answer("induction", None, pairs, None, bad) is False, "x // 2 does not match, expected False"


def test_induction_error_is_incorrect():
    pairs = [(1, 0)]
    got = check_answer("induction", None, pairs, None, "def f(x):\n    return 1 / 0")
    assert got is False, f"a program that raises is incorrect: expected False, got {got}"
    got = check_answer("induction", None, pairs, None, "def g(x):\n    return 0")
    assert got is False, f"a program without f is incorrect: expected False, got {got}"
