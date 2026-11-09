import numpy as np
from submission import accuracy, answer_only, apply_op, make_problem, stepwise


class Fixed:
    """A stand-in for a generator whose random() is always the same number."""

    def __init__(self, x):
        self.x = x
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.x


OPS = [("add", 2), ("mul", 3), ("sub", 4), ("mul", 2)]  # 1 -> 3 -> 9 -> 5 -> 10


def test_answer_only_skips_what_the_budget_cannot_cover():
    for budget, want in [(1, 3), (2, 9), (3, 5), (4, 10), (9, 10)]:
        got = answer_only(1, OPS, budget)
        assert got == want, f"budget {budget}: expected {want}, got {got}"


def test_stepwise_writes_one_value_per_pass():
    values, answer = stepwise(1, OPS, 1)
    assert values == [3, 9, 5, 10] and answer == 10, f"expected ([3, 9, 5, 10], 10), got ({values}, {answer})"
    values, answer = stepwise(1, OPS, 3)
    assert values == [5, 10] and answer == 10, f"budget 3 needs ceil(4/3) = 2 passes; expected ([5, 10], 10), got ({values}, {answer})"
    values, answer = stepwise(1, OPS, 4)
    assert values == [10] and answer == 10, f"expected ([10], 10), got ({values}, {answer})"


def test_stepwise_is_exact_for_any_length_without_noise():
    rng = np.random.default_rng(3)
    for k in range(1, 9):
        for budget in (1, 2, 3):
            start, ops = make_problem(rng, k)
            truth = start
            for op in ops:
                truth = apply_op(truth, op)
            got = stepwise(start, ops, budget)[1]
            assert got == truth, f"k={k}, budget={budget}: expected {truth}, got {got}"


def test_answer_only_is_wrong_beyond_the_budget():
    rng = np.random.default_rng(4)
    for _ in range(50):
        start, ops = make_problem(rng, 3)
        truth = start
        for op in ops:
            truth = apply_op(truth, op)
        assert answer_only(start, ops, 3) == truth, "a budget of 3 covers three operations"
        if truth != 0:  # multiplying 0 by anything leaves it unchanged
            assert answer_only(start, ops, 2) != truth, f"a skipped operation changes the value: {start}, {ops}"


def test_one_draw_per_pass():
    never = Fixed(1.0)
    answer_only(1, OPS, 1, eps=0.5, rng=never)
    assert never.calls == 1, f"answer_only is one pass and should draw once, drew {never.calls}"
    never = Fixed(1.0)
    stepwise(1, OPS, 1, eps=0.5, rng=never)
    assert never.calls == 4, f"4 operations at budget 1 are 4 passes and 4 draws, got {never.calls}"
    never = Fixed(1.0)
    stepwise(1, OPS, 3, eps=0.5, rng=never)
    assert never.calls == 2, f"4 operations at budget 3 are 2 passes and 2 draws, got {never.calls}"


def test_an_error_carries_into_later_passes():
    always = Fixed(0.0)
    values, answer = stepwise(1, [("add", 2), ("mul", 3)], 1, eps=1.0, rng=always)
    assert values == [4, 13] and answer == 13, f"1+2=3, error -> 4; 4*3=12, error -> 13. Expected ([4, 13], 13), got ({values}, {answer})"
    got = answer_only(1, [("add", 2), ("mul", 3)], 1, eps=1.0, rng=Fixed(0.0))
    assert got == 4, f"expected 4 (3 plus the error), got {got}"


def test_no_rng_means_no_noise():
    assert answer_only(1, OPS, 4, eps=1.0) == 10, "without an rng there are no errors"
    assert stepwise(1, OPS, 1, eps=1.0)[1] == 10, "without an rng there are no errors"


def test_accuracy_without_noise():
    got = accuracy([1, 2, 3, 6], budget=2, eps=0.0, n=300, seed=1)
    assert got[1] == (1.0, 1.0), f"k=1 fits the budget; expected (1.0, 1.0), got {got[1]}"
    assert got[2] == (1.0, 1.0), f"k=2 fits the budget; expected (1.0, 1.0), got {got[2]}"
    assert got[3][1] == 1.0 and got[6][1] == 1.0, f"stepwise is exact without noise, got {got[3]} and {got[6]}"
    assert got[3][0] < 0.05, f"one operation over budget; expected answer-only near 0, got {got[3][0]}"
    assert got[6][0] < 0.05, f"expected answer-only near 0 at k=6, got {got[6][0]}"


def test_accuracy_with_noise_matches_the_closed_form():
    eps, budget = 0.1, 1
    got = accuracy([1, 2, 4], budget=budget, eps=eps, n=3000, seed=2)
    for k in (1, 2, 4):
        want_chain = (1 - eps) ** k
        assert abs(got[k][1] - want_chain) < 0.04, f"k={k}: stepwise should be near (1-eps)^k = {want_chain:.3f}, got {got[k][1]:.3f}"
    assert abs(got[1][0] - 0.9) < 0.04, f"k=1 fits one pass; answer-only should be near 0.9, got {got[1][0]:.3f}"
    assert got[2][0] < 0.1, f"k=2 exceeds the budget; answer-only should be near 0, got {got[2][0]:.3f}"


def test_more_passes_per_budget_unit_cost_accuracy():
    eps = 0.15
    got = accuracy([6], budget=3, eps=eps, n=3000, seed=5)
    want = (1 - eps) ** 2
    assert abs(got[6][1] - want) < 0.04, f"k=6 at budget 3 is 2 passes; expected near {want:.3f}, got {got[6][1]:.3f}"
