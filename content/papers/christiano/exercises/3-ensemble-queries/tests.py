import numpy as np
from submission import ensemble_reward, normalize_rewards, select_queries


def test_normalize_known_values():
    got = normalize_rewards(np.array([1.0, 2.0, 3.0]))
    want = np.array([-1.2247, 0.0, 1.2247])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_normalize_moments_on_a_matrix():
    rng = np.random.default_rng(0)
    got = normalize_rewards(rng.normal(3.0, 7.0, size=(5, 8)))
    assert abs(got.mean()) < 1e-9 and abs(got.std() - 1.0) < 1e-9, f"expected mean 0 and std 1, got {got.mean()} and {got.std()}"
    assert got.shape == (5, 8), f"expected shape (5, 8), got {got.shape}"


def test_normalize_constant_input():
    got = normalize_rewards(np.full(4, 2.5))
    assert np.all(np.isfinite(got)) and np.allclose(got, 0), f"expected zeros for a constant input, got {got}"


def test_ensemble_ignores_member_scale():
    rng = np.random.default_rng(1)
    F = rng.normal(size=(6, 5, 3))
    w = np.array([1.0, -1.0, 0.5])
    got = ensemble_reward([w, 10 * w], F)
    want = normalize_rewards(F @ w)
    assert got.shape == (6, 5), f"expected shape (6, 5), got {got.shape}"
    assert np.allclose(got, want), "two members that differ only in scale must average to one normalized reward"


def test_ensemble_normalizes_before_averaging():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(8, 4, 3))
    w1 = np.array([1.0, 0.0, 0.0])
    w2 = np.array([0.0, 50.0, 0.0])
    got = ensemble_reward([w1, w2], F)
    want = 0.5 * (normalize_rewards(F @ w1) + normalize_rewards(F @ w2))
    assert np.allclose(got, want), f"expected the mean of the normalized members, got max error {np.abs(got - want).max():.4f}"


def pairs(f):
    f = np.asarray(f, dtype=float)
    F1 = f.reshape(-1, 1, 1)
    return F1, np.zeros_like(F1)


def test_selects_the_most_contested_pairs_in_order():
    ws = [np.array([1.0]), np.array([-1.0]), np.array([2.0])]
    F1, F2 = pairs([0.0, 3.0, 0.1, 1.0])
    got = list(select_queries(ws, F1, F2, 2))
    assert got == [1, 3], f"expected [1, 3] (highest variance first), got {got}"


def test_uses_variance_across_members_not_pairs():
    ws = [np.array([1.0]), np.array([-1.0])]
    F1, F2 = pairs([2.0, 0.0, 0.0, 2.0, 0.0])
    got = set(select_queries(ws, F1, F2, 2))
    assert got == {0, 3}, f"the two pairs with f = 2 split the members; expected {{0, 3}}, got {got}"


def test_agreement_in_probability_is_not_disagreement():
    ws = [np.array([1.0]), np.array([3.0]), np.array([2.0])]
    F1, F2 = pairs([20.0, 0.2])
    got = list(select_queries(ws, F1, F2, 1))
    assert got == [1], f"all members are certain about pair 0, so expected pair 1, got {got}"


def test_returns_m_indices():
    rng = np.random.default_rng(3)
    ws = [rng.normal(size=4) for _ in range(3)]
    F1, F2 = rng.normal(size=(30, 5, 4)), rng.normal(size=(30, 5, 4))
    got = np.asarray(select_queries(ws, F1, F2, 7))
    assert got.shape == (7,) and len(set(got.tolist())) == 7, f"expected 7 distinct indices, got {got}"
