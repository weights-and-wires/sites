import numpy as np

from submission import device_balance_loss

F = np.array([2.0, 2.0, 0.0, 0.0])
P = np.array([0.55, 0.30, 0.10, 0.05])


def test_contiguous_groups_match_the_page():
    got = device_balance_loss(F, P, [[0, 1], [2, 3]])
    assert abs(got - 1.7) < 1e-9, f"expected 1.7, got {got}"


def test_interleaved_groups_balance_the_devices():
    got = device_balance_loss(F, P, [[0, 2], [1, 3]])
    assert abs(got - 1.0) < 1e-9, f"expected 1.0 (each device holds f' = 1), got {got}"


def test_uneven_groups_use_the_group_mean():
    f = np.array([3.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    p = np.array([0.4, 0.1, 0.1, 0.1, 0.2, 0.1])
    got = device_balance_loss(f, p, [[0, 1, 2], [3, 4, 5]])
    want = (3.0 / 3) * 0.6 + (1.0 / 3) * 0.4
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}"


def test_one_device_per_expert_equals_the_expert_loss():
    got = device_balance_loss(F, P, [[0], [1], [2], [3]])
    want = float((F * P).sum())
    assert abs(got - want) < 1e-9, f"with singleton groups the loss is sum f_j P_j = {want}, got {got}"


def test_alpha_scales_linearly():
    a = device_balance_loss(F, P, [[0, 1], [2, 3]], alpha=1.0)
    b = device_balance_loss(F, P, [[0, 1], [2, 3]], alpha=0.05)
    assert abs(b - 0.05 * a) < 1e-12, f"expected {0.05 * a}, got {b}"
