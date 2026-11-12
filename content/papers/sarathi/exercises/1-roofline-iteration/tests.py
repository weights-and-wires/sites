from submission import arithmetic_intensity, iteration_time, token_budget

P, PEAK, BW = 1e9, 1e14, 1e12  # memory floor 2 ms, 0.02 ms of math per token


def test_intensity_fp16():
    got = arithmetic_intensity(100)
    assert abs(got - 100) < 1e-9, f"2 FLOPs per token per parameter over 2 bytes: expected 100, got {got}"


def test_intensity_one_byte_weights():
    got = arithmetic_intensity(100, 1)
    assert abs(got - 200) < 1e-9, f"with 1-byte weights expected 200, got {got}"


def test_decode_is_memory_bound_and_flat():
    for n in (1, 8, 50):
        t, b = iteration_time(n, P, PEAK, BW)
        assert b == "memory", f"n={n} must be memory-bound, got {b}"
        assert abs(t - 2e-3) < 1e-12, f"n={n}: time must equal the 2 ms weight read, got {t}"


def test_prefill_is_compute_bound_and_linear():
    t, b = iteration_time(200, P, PEAK, BW)
    assert b == "compute", f"n=200 must be compute-bound, got {b}"
    assert abs(t - 4e-3) < 1e-12, f"expected 4 ms, got {t}"
    t2, _ = iteration_time(1000, P, PEAK, BW)
    assert abs(t2 - 2e-2) < 1e-12, f"expected 20 ms for 1000 tokens, got {t2}"


def test_bytes_per_param_changes_the_floor():
    t, b = iteration_time(1, P, PEAK, BW, 1)
    assert b == "memory" and abs(t - 1e-3) < 1e-12, f"1-byte weights: expected (0.001, memory), got {(t, b)}"
    t, b = iteration_time(60, P, PEAK, BW, 1)
    assert b == "compute", f"60 tokens with 1-byte weights is past the ridge (50), got {b}"


def test_token_budget():
    got = token_budget(0.01, P, PEAK, BW)
    assert got == 500, f"10 ms allows 500 tokens, got {got}"
    got = token_budget(0.0105, P, PEAK, BW)
    assert got == 525, f"10.5 ms allows 525 tokens, got {got}"
    assert isinstance(got, int), f"token_budget must return an int, got {type(got)}"


def test_token_budget_below_the_floor():
    got = token_budget(0.001, P, PEAK, BW)
    assert got == 0, f"1 ms is below the 2 ms weight read, expected 0, got {got}"


def test_budget_round_trip():
    n = token_budget(0.0123, P, PEAK, BW)
    t, _ = iteration_time(n, P, PEAK, BW)
    t_next, _ = iteration_time(n + 1, P, PEAK, BW)
    assert t <= 0.0123 + 1e-12 < t_next, f"n={n} must fit and n+1 must not; got {t} and {t_next}"
