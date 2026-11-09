import numpy as np
from submission import attainable, is_memory_bound, ridge_point

PEAK, BW = 17.6, 15.0  # Opteron X2, Section 3


def test_memory_bound_value():
    got = float(attainable(1.0, PEAK, BW))
    assert np.isclose(got, 15.0), f"intensity 1 on the X2: expected 15.0 GFLOP/s, got {got}"


def test_compute_bound_value():
    got = float(attainable(2.0, PEAK, BW))
    assert np.isclose(got, 17.6), f"intensity 2 on the X2: expected 17.6 GFLOP/s, got {got}"


def test_array_input():
    oi = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    want = np.array([3.75, 7.5, 15.0, 17.6, 17.6])
    got = attainable(oi, PEAK, BW)
    assert np.shape(got) == oi.shape, f"expected shape {oi.shape}, got {np.shape(got)}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_never_above_either_roof():
    oi = np.logspace(-2, 3, 50)
    got = attainable(oi, 100.0, 10.0)
    assert np.all(got <= 100.0 + 1e-9), f"attainable exceeds the compute roof: max {got.max()}"
    assert np.all(got <= 10.0 * oi + 1e-9), "attainable exceeds the bandwidth roof"
    assert np.isclose(got[0], 10.0 * oi[0]) and np.isclose(got[-1], 100.0), f"ends should sit on each roof, got {got[0]} and {got[-1]}"


def test_ridge_point():
    got = ridge_point(PEAK, BW)
    assert np.isclose(got, 17.6 / 15.0), f"X2 ridge expected {17.6 / 15.0:.4f}, got {got}"
    got4 = ridge_point(73.6, 16.6)
    assert np.isclose(got4, 73.6 / 16.6), f"X4 ridge expected {73.6 / 16.6:.4f}, got {got4}"


def test_ridge_is_where_roofs_meet():
    r = ridge_point(250.0, 40.0)
    assert np.isclose(float(attainable(r, 250.0, 40.0)), 250.0), f"at the ridge the roof should equal the peak 250, got {attainable(r, 250.0, 40.0)}"
    assert np.isclose(40.0 * r, 250.0), f"bw * ridge should equal peak, got {40.0 * r}"


def test_regime():
    assert bool(is_memory_bound(1.0, PEAK, BW)) is True, "intensity 1 on the X2 should be memory-bound"
    assert bool(is_memory_bound(2.0, PEAK, BW)) is False, "intensity 2 on the X2 should be compute-bound"
    r = ridge_point(PEAK, BW)
    assert bool(is_memory_bound(r, PEAK, BW)) is False, "a kernel exactly at the ridge point is not memory-bound"
    assert bool(is_memory_bound(r * 0.99, PEAK, BW)) is True, "just left of the ridge point is memory-bound"
