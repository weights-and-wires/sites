from submission import pretrain_k_hours, days_per_trillion, total_k_hours, cost_million_usd, stage_shares

STAGES = [2664, 119, 5]


def test_pretrain_hours():
    got = pretrain_k_hours(180, 14.8)
    assert abs(got - 2664) < 1e-6, f"expected 180 x 14.8 = 2664 (Table 1), got {got}"


def test_days_per_trillion():
    got = days_per_trillion(180, 2048)
    assert abs(got - 3.6621) < 1e-3, f"expected 180000 / 2048 / 24 = 3.662 days (Section 1 says 3.7), got {got}"


def test_pretrain_days_under_two_months():
    days = days_per_trillion(180, 2048) * 14.8
    assert abs(days - 54.2) < 0.1, f"expected about 54.2 days, got {days}"


def test_total():
    got = total_k_hours(STAGES)
    assert got == 2788, f"expected 2788 (Table 1), got {got}"


def test_cost_matches_table_1():
    got = cost_million_usd(2788, 2.0)
    assert abs(got - 5.576) < 1e-9, f"expected $5.576M, got {got}"
    got2 = cost_million_usd(5, 2.0)
    assert abs(got2 - 0.01) < 1e-9, f"expected $0.01M for 5K hours, got {got2}"


def test_cost_scales_with_price():
    got = cost_million_usd(1000, 3.0)
    assert abs(got - 3.0) < 1e-9, f"1000K hours at $3 is $3M, got {got}"


def test_shares():
    got = stage_shares(STAGES)
    want = [2664 / 2788, 119 / 2788, 5 / 2788]
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, want)), f"expected {want}, got {got}"
    assert abs(sum(got) - 1.0) < 1e-12, f"shares must sum to 1, got {sum(got)}"
