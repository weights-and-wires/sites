from submission import composite_reward


def test_solver_cases():
    cases = [((1.0, True, True), 1.0), ((0.0, True, False), -0.5), ((1.0, False, True), -1.0)]
    for args, want in cases:
        got = composite_reward(*args)
        assert got == want, f"composite_reward{args}: expected {want}, got {got}"


def test_proposer_keeps_fractional_reward():
    got = composite_reward(0.75, True, True)
    assert abs(got - 0.75) < 1e-9, f"a valid task keeps its learnability reward: expected 0.75, got {got}"


def test_valid_task_with_zero_learnability_is_zero_not_penalty():
    got = composite_reward(0.0, True, True)
    assert got == 0.0, f"a valid task that earns 0 is not penalized: expected 0.0, got {got}"


def test_format_error_beats_everything():
    for passed in (True, False):
        got = composite_reward(0.9, False, passed)
        assert got == -1.0, f"formatting error with passed={passed}: expected -1.0, got {got}"


def test_invalid_task_ignores_role_reward():
    got = composite_reward(0.9, True, False)
    assert got == -0.5, f"a well-formed failure earns -0.5 whatever r_role is: expected -0.5, got {got}"


def test_returns_float():
    got = composite_reward(1, True, True)
    assert isinstance(got, float), f"expected a float, got {type(got)}"
