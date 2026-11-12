import numpy as np
from submission import task_relative_advantages

R = [1, 0, 0, 1, 0.75, 0.5, 0.125, 0.0, 1, 1, 0, 1]
T = ["ded"] * 8 + ["abd"] * 4
O = ["solve"] * 4 + ["propose"] * 4 + ["solve"] * 4


def test_page_example():
    a = task_relative_advantages(R, T, O)
    assert a.shape == (12,), f"expected shape (12,), got {a.shape}"
    want = np.array([1, -1, -1, 1], dtype=float)
    assert np.allclose(a[:4], want, atol=1e-4), f"ded/solve: expected {want}, got {a[:4]}"
    want = np.array([1.3628, 0.5241, -0.7338, -1.1531])
    assert np.allclose(a[4:8], want, atol=1e-3), f"ded/propose: expected {want}, got {a[4:8]}"
    want = np.array([0.5773, 0.5773, -1.732, 0.5773])
    assert np.allclose(a[8:], want, atol=1e-3), f"abd/solve: expected {want}, got {a[8:]}"


def test_each_group_has_mean_zero():
    a = task_relative_advantages(R, T, O)
    for sl in (slice(0, 4), slice(4, 8), slice(8, 12)):
        assert abs(a[sl].mean()) < 1e-8, f"group mean must be 0, got {a[sl].mean()} for {a[sl]}"


def test_roles_are_separate_baselines():
    # Same task type, different roles: pooling them changes the answer.
    r = [1, 0, 0.2, 0.1]
    a = task_relative_advantages(r, ["ded"] * 4, ["solve", "solve", "propose", "propose"])
    assert np.allclose(a[:2], [1, -1], atol=1e-4), f"solve pair: expected [1, -1], got {a[:2]}"
    assert np.allclose(a[2:], [1, -1], atol=1e-4), f"propose pair: expected [1, -1], got {a[2:]}"


def test_task_types_are_separate_baselines():
    r = [1, 0, 10, 0]
    a = task_relative_advantages(r, ["ded", "ded", "abd", "abd"], ["solve"] * 4)
    assert np.allclose(a, [1, -1, 1, -1], atol=1e-4), f"expected [1, -1, 1, -1], got {a}"


def test_order_is_preserved_when_interleaved():
    r = [1, 5, 0, 5, 1, 9]
    t = ["ded", "abd", "ded", "abd", "ded", "abd"]
    a = task_relative_advantages(r, t, ["solve"] * 6)
    want_ded = task_relative_advantages([1, 0, 1], ["ded"] * 3, ["solve"] * 3)
    assert np.allclose(a[[0, 2, 4]], want_ded, atol=1e-6), f"expected {want_ded} at positions 0, 2, 4, got {a[[0, 2, 4]]}"


def test_constant_group_gives_zero_not_nan():
    a = task_relative_advantages([1, 1, 1, 0, 1], ["ded"] * 5, ["solve"] * 3 + ["propose"] * 2)
    assert np.all(np.isfinite(a)), f"expected finite values, got {a}"
    assert np.allclose(a[:3], 0, atol=1e-6), f"a constant group must get 0, got {a[:3]}"


def test_population_std():
    a = task_relative_advantages([1.0, 0.0], ["ind", "ind"], ["solve", "solve"])
    assert np.allclose(a, [1, -1], atol=1e-4), f"std of [1, 0] is 0.5 (ddof=0): expected [1, -1], got {a}"
