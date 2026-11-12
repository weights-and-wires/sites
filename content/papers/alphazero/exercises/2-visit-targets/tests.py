import numpy as np
from submission import visits_to_policy

N = np.array([60, 30, 8, 2])


def test_tau_one_is_the_visit_share():
    got = visits_to_policy(N, 1.0)
    want = np.array([0.6, 0.3, 0.08, 0.02])
    assert np.allclose(got, want, atol=1e-9), f"expected {want}, got {got}"


def test_tau_half_squares_the_counts():
    got = visits_to_policy(N, 0.5)
    want = np.array([0.7881, 0.1970, 0.0140, 0.0009])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {np.round(got, 4)}"


def test_tau_two_is_flatter_than_tau_one():
    got = visits_to_policy(N, 2.0)
    want = np.array([0.4435, 0.3136, 0.1619, 0.0810])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {np.round(got, 4)}"


def test_target_sums_to_one_and_keeps_zero_counts_at_zero():
    for tau in (0.0, 0.3, 1.0, 5.0):
        got = visits_to_policy([10, 0, 5, 0], tau)
        assert abs(got.sum() - 1.0) < 1e-9, f"tau = {tau}: expected the target to sum to 1, got {got.sum()}"
        assert got[1] == 0 and got[3] == 0, f"tau = {tau}: unvisited moves must stay at 0, got {got}"


def test_small_tau_does_not_overflow():
    got = visits_to_policy(N, 0.001)
    assert np.all(np.isfinite(got)), f"expected finite values at tau = 0.001 (work in log space), got {got}"
    assert np.allclose(got, [1, 0, 0, 0], atol=1e-9), f"expected [1, 0, 0, 0], got {got}"
    got = visits_to_policy([800, 790], 0.001)
    assert np.all(np.isfinite(got)), f"expected finite values for counts near 800, got {got}"
    assert got[0] > 0.999, f"expected nearly all mass on the larger count, got {got}"


def test_tau_zero_is_greedy_and_splits_ties():
    got = visits_to_policy(N, 0.0)
    assert np.array_equal(got, [1, 0, 0, 0]), f"expected [1, 0, 0, 0], got {got}"
    got = visits_to_policy([5, 5, 0], 0.0)
    assert np.allclose(got, [0.5, 0.5, 0.0]), f"expected the tie to split as [0.5, 0.5, 0], got {got}"


def test_lower_tau_is_sharper():
    top = [visits_to_policy(N, t)[0] for t in (4.0, 2.0, 1.0, 0.5, 0.25, 0.1)]
    assert all(a < b for a, b in zip(top, top[1:])), f"expected the top share to rise as tau falls, got {np.round(top, 4)}"


def test_input_list_and_input_not_modified():
    counts = np.array([3.0, 1.0])
    before = counts.copy()
    visits_to_policy(counts, 1.0)
    assert np.array_equal(counts, before), f"expected N unchanged, got {counts}"
    got = visits_to_policy([3, 1], 1.0)
    assert np.allclose(got, [0.75, 0.25]), f"expected [0.75, 0.25] from a plain list, got {got}"
