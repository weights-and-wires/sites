import math

from submission import avg_ttft, avg_ttft_inter, avg_ttft_intra, better_parallelism


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_eq1_values():
    got = avg_ttft(0.5, 1.0)
    assert close(got, 1.5), f"R D = 0.5, D = 1: 1 + 0.5 / (2 * 0.5) = 1.5, got {got}"
    got = avg_ttft(0.8, 1.0)
    assert close(got, 3.0), f"R D = 0.8, D = 1: 1 + 0.8 / 0.4 = 3.0, got {got}"


def test_eq1_scales_with_d():
    got = avg_ttft(2.0, 0.2)
    want = 0.2 + 2.0 * 0.04 / (2 * 0.6)
    assert close(got, want), f"expected {want}, got {got}"


def test_eq1_unstable():
    got = avg_ttft(1.0, 1.0)
    assert got == math.inf, f"R D = 1 is unstable, expected inf, got {got}"
    got = avg_ttft(2.0, 1.0)
    assert got == math.inf, f"R D = 2 is unstable, expected inf, got {got}"


def test_eq1_idle_queue_is_execution_time():
    got = avg_ttft(0.0, 0.3)
    assert close(got, 0.3), f"with no arrivals the TTFT is D = 0.3, got {got}"


def test_eq2_values():
    got = avg_ttft_inter(1.0, 1.0)
    assert close(got, 1.25), f"1 + 1 / (4 * 1) = 1.25, got {got}"
    got = avg_ttft_inter(0.5, 1.0)
    want = 1 + 0.5 / (4 * 1.5)
    assert close(got, want), f"expected {want}, got {got}"


def test_eq2_stays_stable_past_one_gpu_capacity():
    got = avg_ttft_inter(1.5, 1.0)
    assert close(got, 1.75), f"R D = 1.5 is stable with two stages: 1 + 1.5 / 2 = 1.75, got {got}"
    got = avg_ttft_inter(2.0, 1.0)
    assert got == math.inf, f"R D = 2 is unstable, expected inf, got {got}"


def test_eq3_values():
    got = avg_ttft_intra(0.5, 1.0, 1.6)
    want = 1 / 1.6 + 0.5 / (2 * 1.6 * 1.1)
    assert close(got, want), f"expected {want}, got {got}"
    got = avg_ttft_intra(0.0, 1.0, 2.0)
    assert close(got, 0.5), f"with no arrivals the TTFT is D / K = 0.5, got {got}"


def test_eq3_limit_is_k():
    got = avg_ttft_intra(1.5, 1.0, 1.6)
    assert math.isfinite(got), f"R D = 1.5 < K = 1.6 is stable, got {got}"
    got = avg_ttft_intra(1.6, 1.0, 1.6)
    assert got == math.inf, f"R D = K is unstable, expected inf, got {got}"


def test_intra_wins_at_low_rate():
    for k in (1.5, 1.9):
        got = better_parallelism(0.3, 1.0, k)
        assert got == "intra", f"K = {k}, R D = 0.3: execution time dominates, expected intra, got {got}"


def test_inter_wins_at_high_rate():
    got = better_parallelism(1.3, 1.0, 1.6)
    assert got == "inter", f"K = 1.6, R D = 1.3: intra is 1.979 and inter is 1.464, expected inter, got {got}"


def test_crossover_depends_on_k():
    got = better_parallelism(1.1, 1.0, 1.6)
    assert got == "inter", f"K = 1.6, R D = 1.1: intra 1.312 vs inter 1.306, expected inter, got {got}"
    got = better_parallelism(1.1, 1.0, 1.9)
    assert got == "intra", f"K = 1.9, R D = 1.1: intra 0.888 vs inter 1.306, expected intra, got {got}"
