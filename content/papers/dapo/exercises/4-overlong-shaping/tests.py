from submission import soft_overlong_penalty, total_reward


def close(a, b):
    return abs(a - b) < 1e-9


def test_no_penalty_up_to_the_start_of_the_ramp():
    for n in (1, 8000, 16384):
        got = soft_overlong_penalty(n)
        assert close(got, 0.0), f"length {n}: expected 0.0, got {got}"


def test_ramp_values_from_the_page():
    cases = {18432: -0.5, 20000: -0.8828125, 20480: -1.0, 16385: -1 / 4096}
    for n, want in cases.items():
        got = soft_overlong_penalty(n)
        assert close(got, want), f"length {n}: expected {want}, got {got}"


def test_beyond_the_limit_is_minus_one():
    for n in (20481, 25000):
        got = soft_overlong_penalty(n)
        assert close(got, -1.0), f"length {n}: expected -1.0, got {got}"


def test_penalty_is_monotone_non_increasing():
    prev = 1.0
    for n in range(0, 22000, 97):
        cur = soft_overlong_penalty(n)
        assert cur <= prev + 1e-12, f"penalty rose from {prev} to {cur} at length {n}"
        prev = cur


def test_custom_constants():
    got = soft_overlong_penalty(75, l_max=100, l_cache=50)
    assert close(got, -0.5), f"expected (50 - 75) / 50 = -0.5, got {got}"
    got = soft_overlong_penalty(50, l_max=100, l_cache=50)
    assert close(got, 0.0), f"expected 0.0 at the ramp start, got {got}"


def test_total_reward_adds_the_penalty():
    cases = [(True, 16000, 1.0), (True, 18432, 0.5), (True, 20480, 0.0), (False, 18432, -1.5), (False, 100, -1.0)]
    for correct, n, want in cases:
        got = total_reward(correct, n)
        assert close(got, want), f"correct={correct}, length={n}: expected {want}, got {got}"


def test_total_reward_passes_constants_through():
    got = total_reward(True, 75, l_max=100, l_cache=50)
    assert close(got, 0.5), f"expected 1 - 0.5 = 0.5, got {got}"
