import numpy as np
from submission import returns_to_go


def test_worked_example():
    got = returns_to_go([1, 0, 2, -1])
    want = np.array([2, 1, 1, -1.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_sparse_reward_reaches_every_step():
    got = returns_to_go([0, 0, 0, 1])
    want = np.array([1, 1, 1, 1.0])
    assert np.allclose(got, want), f"every step before the reward must carry it: expected {want}, got {got}"


def test_first_entry_is_total_and_last_is_final_reward():
    r = np.array([3.0, -2.0, 5.0, 0.5, 1.5])
    g = returns_to_go(r)
    assert np.isclose(g[0], r.sum()), f"g[0] expected {r.sum()}, got {g[0]}"
    assert np.isclose(g[-1], r[-1]), f"g[-1] expected {r[-1]}, got {g[-1]}"


def test_current_reward_is_included():
    g = returns_to_go([5, 0, 0])
    assert np.allclose(g, [5, 0, 0]), f"expected [5, 0, 0], got {g}"


def test_recursion_g_t_equals_r_t_plus_g_next():
    rng = np.random.default_rng(1)
    r = rng.normal(size=12)
    g = returns_to_go(r)
    assert np.allclose(g[:-1], r[:-1] + g[1:]), f"expected g[t] = r[t] + g[t+1]; max gap {np.abs(g[:-1] - r[:-1] - g[1:]).max()}"


def test_batch_runs_along_time_axis():
    r = np.array([[1, 0, 2, -1], [0, 0, 0, 1], [2, 2, 2, 2.0]])
    want = np.array([[2, 1, 1, -1], [1, 1, 1, 1], [8, 6, 4, 2.0]])
    got = returns_to_go(r)
    assert got.shape == r.shape, f"expected shape {r.shape}, got {got.shape}"
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_input_unchanged_and_float_output():
    r = np.array([1, 2, 3])
    before = r.copy()
    g = returns_to_go(r)
    assert np.array_equal(r, before), f"input was modified: {before} -> {r}"
    assert np.issubdtype(g.dtype, np.floating), f"expected a float array, got dtype {g.dtype}"
    assert np.allclose(g, [6, 5, 3]), f"expected [6, 5, 3], got {g}"
