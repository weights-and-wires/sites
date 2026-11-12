from submission import max_goodput, slo_attainment


def test_both_slos_must_hold():
    ttft = [0.1, 0.5, 0.2, 0.2]
    tpot = [0.03, 0.03, 0.06, 0.04]
    got = slo_attainment(ttft, tpot, 0.4, 0.05)
    assert got == 0.5, f"requests 0 and 3 meet both SLOs, expected 0.5, got {got}"


def test_one_slo_alone_is_not_enough():
    ttft = [0.1, 0.1]
    tpot = [0.9, 0.9]
    got = slo_attainment(ttft, tpot, 0.4, 0.05)
    assert got == 0.0, f"TTFT is fine but TPOT misses for both, expected 0.0, got {got}"
    got = slo_attainment(tpot, ttft, 0.4, 0.05)
    assert got == 0.0, f"TPOT is fine but TTFT misses for both, expected 0.0, got {got}"


def test_boundary_counts_as_met():
    got = slo_attainment([0.4], [0.05], 0.4, 0.05)
    assert got == 1.0, f"a request exactly at both SLOs meets them, expected 1.0, got {got}"


def test_all_good_and_all_bad():
    got = slo_attainment([0.1] * 5, [0.01] * 5, 1.0, 1.0)
    assert got == 1.0, f"expected 1.0, got {got}"
    got = slo_attainment([2.0] * 5, [2.0] * 5, 1.0, 1.0)
    assert got == 0.0, f"expected 0.0, got {got}"


def test_attainment_divides_by_all_requests():
    got = slo_attainment([0.1, 9, 9, 9, 9, 9, 9, 9, 9, 9], [0.01] * 10, 1.0, 1.0)
    assert abs(got - 0.1) < 1e-12, f"1 of 10 requests is good, expected 0.1, got {got}"


def test_bisection_finds_a_step():
    f = lambda rate: 1.0 if rate <= 3.7 else 0.4
    got = max_goodput(f)
    assert abs(got - 3.7) < 1e-6, f"the attainment drops after 3.7, expected 3.7, got {got}"
    assert f(got) >= 0.9, f"the answer must meet the target, attainment at {got} is {f(got)}"


def test_bisection_on_a_smooth_curve():
    f = lambda rate: 1.0 - rate / 10
    got = max_goodput(f, target=0.9)
    assert abs(got - 1.0) < 1e-6, f"1 - rate/10 reaches 0.9 at rate 1.0, got {got}"
    got = max_goodput(f, target=0.5)
    assert abs(got - 5.0) < 1e-6, f"the 0.5 target is met up to rate 5.0, got {got}"


def test_answer_always_meets_the_target():
    f = lambda rate: 1.0 if rate < 2.0 else 0.0
    got = max_goodput(f, iters=10)
    assert f(got) >= 0.9, f"with few rounds the answer must still meet the target, got {got} with attainment {f(got)}"
    assert 1.9 < got < 2.0, f"10 rounds on [0, 100] narrow 2.0 to within 0.1, got {got}"


def test_search_range_is_respected():
    f = lambda rate: 1.0
    got = max_goodput(f, lo=1.0, hi=8.0, iters=40)
    assert abs(got - 8.0) < 1e-6, f"every rate meets the target, so the search ends at hi = 8.0, got {got}"


def test_target_is_inclusive():
    f = lambda rate: 0.9 if rate <= 2.0 else 0.5
    got = max_goodput(f, target=0.9)
    assert abs(got - 2.0) < 1e-6, f"attainment of exactly 0.9 meets the 90% goal, so the answer is 2.0, got {got}"
