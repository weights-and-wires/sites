import math

from submission import layout


def test_conventional_top2_of_16():
    r = layout(16, 2)
    assert r["combinations"] == 120, f"expected C(16,2) = 120, got {r['combinations']}"
    assert r["n_routed"] == 16 and r["k_active"] == 2, f"expected 16 routed and 2 active, got {r['n_routed']} and {r['k_active']}"


def test_fine_grained_matches_the_paper():
    r = layout(16, 2, m=4)
    assert r["n_routed"] == 64 and r["k_active"] == 8, f"expected 64 routed and 8 active, got {r['n_routed']} and {r['k_active']}"
    assert r["combinations"] == 4_426_165_368, f"expected 4,426,165,368, got {r['combinations']}"


def test_shared_expert_reduces_routed_counts():
    r = layout(16, 2, m=4, Ks=1)
    assert r["n_routed"] == 63 and r["k_active"] == 7, f"expected 63 routed and 7 active, got {r['n_routed']} and {r['k_active']}"
    assert r["combinations"] == math.comb(63, 7), f"expected C(63,7) = 553270671, got {r['combinations']}"


def test_expert_size_is_one_over_m():
    assert layout(16, 2, m=4)["expert_size"] == 0.25, f"expected 0.25, got {layout(16, 2, m=4)['expert_size']}"
    assert layout(16, 2)["expert_size"] == 1.0, f"expected 1.0, got {layout(16, 2)['expert_size']}"


def test_budget_is_constant():
    for m, ks in [(1, 0), (2, 0), (4, 0), (4, 1), (8, 2)]:
        r = layout(16, 2, m=m, Ks=ks)
        assert abs(r["total_ffn"] - 16) < 1e-9, f"m={m}, Ks={ks}: expected total_ffn 16, got {r['total_ffn']}"
        assert abs(r["active_ffn"] - 2) < 1e-9, f"m={m}, Ks={ks}: expected active_ffn 2, got {r['active_ffn']}"


def test_other_sizes():
    r = layout(8, 2, m=2, Ks=1)
    assert (r["n_routed"], r["k_active"], r["combinations"]) == (15, 3, 455), f"expected (15, 3, 455), got {(r['n_routed'], r['k_active'], r['combinations'])}"
