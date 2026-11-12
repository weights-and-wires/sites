from submission import learnability_reward


def test_page_values():
    for k, want in [(1, 0.875), (2, 0.75), (4, 0.5), (6, 0.25), (7, 0.125)]:
        got = learnability_reward([1] * k + [0] * (8 - k))
        assert abs(got - want) < 1e-9, f"{k} successes of 8: expected {want}, got {got}"


def test_never_solved_is_zero():
    got = learnability_reward([0, 0, 0, 0, 0, 0, 0, 0])
    assert got == 0.0, f"a task the solver never solves must get 0.0, got {got}"


def test_always_solved_is_zero():
    got = learnability_reward([1, 1, 1, 1, 1, 1, 1, 1])
    assert got == 0.0, f"a task the solver always solves must get 0.0, got {got}"


def test_harder_task_earns_more():
    easy = learnability_reward([1, 1, 1, 0])
    hard = learnability_reward([1, 0, 0, 0])
    assert hard > easy, f"expected the harder task to earn more, got hard={hard}, easy={easy}"
    assert abs(hard - 0.75) < 1e-9 and abs(easy - 0.25) < 1e-9, f"expected 0.75 and 0.25, got {hard} and {easy}"


def test_returns_float_and_accepts_lists():
    got = learnability_reward([1, 0])
    assert isinstance(got, float), f"expected a float, got {type(got)}"
    assert abs(got - 0.5) < 1e-9, f"expected 0.5, got {got}"
