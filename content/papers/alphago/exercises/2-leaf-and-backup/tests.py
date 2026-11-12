import numpy as np
from submission import backup, leaf_value, mixed_q


def test_leaf_value_on_page_example():
    got = leaf_value(0.3, -1.0, 0.5)
    assert np.isclose(got, -0.35), f"expected -0.35, got {got}"


def test_leaf_value_endpoints():
    assert np.isclose(leaf_value(0.3, -1.0, 0.0), 0.3), f"lambda = 0 must return the value network output 0.3, got {leaf_value(0.3, -1.0, 0.0)}"
    assert np.isclose(leaf_value(0.3, -1.0, 1.0), -1.0), f"lambda = 1 must return the rollout outcome -1.0, got {leaf_value(0.3, -1.0, 1.0)}"


def test_leaf_value_weights_are_not_swapped():
    got = leaf_value(1.0, 0.0, 0.25)
    assert np.isclose(got, 0.75), f"lambda weighs z, so 0.75 * 1 + 0.25 * 0 = 0.75, got {got}"


def test_backup_on_page_example():
    N = np.zeros(2)
    W = np.zeros(2)
    for v in (0.4, -0.2, 0.6):
        backup(N, W, [0, 1], v)
    assert np.allclose(N, [3, 3]), f"expected N = [3, 3], got {N}"
    assert np.allclose(W, [0.8, -0.8]), f"expected W = [0.8, -0.8], got {W}"
    q = W / N
    assert np.allclose(q, [0.2667, -0.2667], atol=1e-3), f"expected Q = [0.2667, -0.2667], got {q}"


def test_backup_signs_alternate_from_the_leaf():
    N = np.zeros(4)
    W = np.zeros(4)
    backup(N, W, [0, 1, 2, 3], 1.0)
    assert np.allclose(W, [1.0, -1.0, 1.0, -1.0]), f"the last edge gets -v and signs alternate upward, expected W = [1, -1, 1, -1], got {W}"


def test_backup_odd_and_even_path_lengths():
    N = np.zeros(3)
    W = np.zeros(3)
    backup(N, W, [0, 1, 2], 0.5)
    assert np.allclose(W, [-0.5, 0.5, -0.5]), f"expected W = [-0.5, 0.5, -0.5], got {W}"
    N = np.zeros(2)
    W = np.zeros(2)
    backup(N, W, [0, 1], 0.5)
    assert np.allclose(W, [0.5, -0.5]), f"expected W = [0.5, -0.5], got {W}"


def test_backup_touches_only_path_edges_and_updates_in_place():
    N = np.array([5.0, 5.0, 5.0, 5.0])
    W = np.array([1.0, 1.0, 1.0, 1.0])
    N0, W0 = N, W
    backup(N, W, [0, 2], 1.0)
    assert N is N0 and W is W0, "backup must modify the arrays it receives"
    assert np.allclose(N, [6, 5, 6, 5]), f"expected N = [6, 5, 6, 5], got {N}"
    assert np.allclose(W, [2.0, 1.0, 0.0, 1.0]), f"expected W = [2, 1, 0, 1], got {W}"


def test_backup_single_edge_path():
    N = np.zeros(1)
    W = np.zeros(1)
    backup(N, W, [0], 0.7)
    assert np.allclose(W, [-0.7]) and np.allclose(N, [1]), f"the only edge leads into the leaf, expected W = [-0.7], N = [1], got W = {W}, N = {N}"


def test_mixed_q_on_page_example():
    got = mixed_q(1.2, 4, -1.0, 3, 0.5)
    assert np.isclose(got, -0.016667, atol=1e-4), f"expected -0.0167, got {got}"


def test_mixed_q_equals_mean_of_leaf_values_when_counts_match():
    vs = [0.3, 0.1, -0.5, 0.7]
    zs = [-1.0, 1.0, -1.0, 1.0]
    lam = 0.5
    want = np.mean([leaf_value(v, z, lam) for v, z in zip(vs, zs)])
    got = mixed_q(sum(vs), 4, sum(zs), 4, lam)
    assert np.isclose(got, want), f"expected {want}, got {got}"


def test_mixed_q_handles_empty_counts():
    got = mixed_q(0.0, 0, -2.0, 4, 0.5)
    assert np.isclose(got, -0.25), f"with N_v = 0 only the rollout term remains: 0.5 * (-0.5) = -0.25, got {got}"
    got = mixed_q(0.0, 0, 0.0, 0, 0.5)
    assert np.isclose(got, 0.0), f"two empty counts give 0, got {got}"


def test_mixed_q_uses_lambda():
    got = mixed_q(2.0, 2, -2.0, 2, 0.25)
    assert np.isclose(got, 0.5), f"0.75 * 1.0 + 0.25 * (-1.0) = 0.5, got {got}"
    got = mixed_q(2.0, 2, -2.0, 2, 0.0)
    assert np.isclose(got, 1.0), f"lambda = 0 uses only the value network, expected 1.0, got {got}"
