from submission import simulate


def toy(n):
    return 10.0 * max(1.0, n / 100)  # 10 ms floor, ridge at 100 tokens


def lin(n):
    return float(n)


def close(a, b):
    return abs(a - b) < 1e-9


def test_prefill_first_stalls():
    tbt, ttft = simulate("prefill_first", 4, 12, 1000, 3, 0, toy)
    assert close(tbt, 110.0) and close(ttft, 100.0), f"expected (110.0, 100.0), got {(tbt, ttft)}"


def test_stall_free_bounds_tbt():
    tbt, ttft = simulate("stall_free", 4, 12, 1000, 3, 200, toy)
    assert close(tbt, 20.0) and close(ttft, 110.0), f"expected (20.0, 110.0), got {(tbt, ttft)}"


def test_budget_at_the_ridge_is_free():
    tbt, ttft = simulate("stall_free", 4, 12, 1000, 3, 100, toy)
    assert close(tbt, 10.0) and close(ttft, 110.0), f"expected (10.0, 110.0), got {(tbt, ttft)}"


def test_huge_budget_degenerates_to_one_big_iteration():
    tbt, ttft = simulate("stall_free", 4, 12, 1000, 3, 1000, toy)
    assert close(tbt, 100.0) and close(ttft, 110.0), f"expected (100.0, 110.0), got {(tbt, ttft)}"


def test_short_prompt_hardly_matters():
    a = simulate("prefill_first", 4, 12, 50, 3, 0, toy)
    b = simulate("stall_free", 4, 12, 50, 3, 200, toy)
    assert close(a[0], 20.0) and close(a[1], 10.0), f"prefill_first expected (20.0, 10.0), got {a}"
    assert close(b[0], 10.0) and close(b[1], 10.0), f"stall_free expected (10.0, 10.0), got {b}"


def test_linear_cost_hand_check_prefill_first():
    got = simulate("prefill_first", 2, 4, 10, 1, 0, lin)
    assert close(got[0], 12.0) and close(got[1], 10.0), f"expected (12.0, 10.0), got {got}"


def test_linear_cost_hand_check_stall_free():
    got = simulate("stall_free", 2, 4, 10, 1, 6, lin)
    assert close(got[0], 6.0) and close(got[1], 16.0), f"expected (6.0, 16.0), got {got}"


def test_stall_free_beats_prefill_first_on_tbt_and_loses_on_ttft():
    pf = simulate("prefill_first", 8, 30, 2000, 4, 0, toy)
    sf = simulate("stall_free", 8, 30, 2000, 4, 256, toy)
    assert sf[0] < pf[0], f"stall-free max TBT must be lower: {sf[0]} vs {pf[0]}"
    assert sf[1] > pf[1], f"stall-free TTFT must be higher here: {sf[1]} vs {pf[1]}"
