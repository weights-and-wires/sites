import numpy as np

from submission import expert_balance_loss

BAL_S = np.full((4, 4), 0.25)
BAL_IDX = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])
COL_S = np.tile([0.55, 0.30, 0.10, 0.05], (4, 1))
COL_IDX = np.tile([0, 1], (4, 1))


def test_balanced_routing_gives_alpha():
    got = expert_balance_loss(BAL_S, BAL_IDX, alpha=1.0)
    assert abs(got - 1.0) < 1e-9, f"expected 1.0 for balanced routing, got {got}"
    got = expert_balance_loss(BAL_S, BAL_IDX, alpha=0.01)
    assert abs(got - 0.01) < 1e-12, f"expected alpha = 0.01, got {got}"


def test_collapsed_routing_matches_the_page():
    got = expert_balance_loss(COL_S, COL_IDX, alpha=1.0)
    assert abs(got - 1.7) < 1e-9, f"expected 1.7 for the collapsed case on the page, got {got}"


def test_alpha_scales_linearly():
    a = expert_balance_loss(COL_S, COL_IDX, alpha=1.0)
    b = expert_balance_loss(COL_S, COL_IDX, alpha=0.25)
    assert abs(b - 0.25 * a) < 1e-12, f"expected {0.25 * a}, got {b}"


def test_normalization_uses_k_and_n():
    # 6 tokens, N' = 6 experts, K' = 3; balanced counts give f_i = 1 for any K'.
    rng = np.random.default_rng(0)
    T, n, k = 6, 6, 3
    idx = np.stack([(np.arange(T) + j) % n for j in range(k)], axis=1)
    s = rng.dirichlet(np.ones(n), size=T)
    f_counts = np.bincount(idx.ravel(), minlength=n)
    assert (f_counts == k).all(), "test setup is wrong: every expert should receive 3 tokens"
    got = expert_balance_loss(s, idx)
    want = float(s.mean(axis=0).sum())
    assert abs(got - want) < 1e-9, f"expected {want} (f_i = 1 for every expert), got {got}"


def test_p_uses_all_affinities_not_only_selected():
    s = np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
    idx = np.array([[0], [2]])
    got = expert_balance_loss(s, idx)
    f = np.array([1, 0, 1]) * 3 / (1 * 2)
    p = s.mean(axis=0)
    want = float((f * p).sum())
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}"


def test_collapse_costs_more_than_balance():
    assert expert_balance_loss(COL_S, COL_IDX) > expert_balance_loss(BAL_S, BAL_IDX), "a collapsed router must have a larger loss than a balanced one"
