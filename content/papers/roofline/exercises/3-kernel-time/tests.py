import math
from submission import place

PEAK, BW = 100e9, 10e9  # ridge point 10 FLOPs/byte
KERNELS = [
    ("stream", 1e9, 1e9),    # intensity 1
    ("ridge", 1e10, 1e9),    # intensity 10
    ("matmul", 1e11, 1e9),   # intensity 100
]


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9)


def test_order_and_names():
    got = [r["name"] for r in place(KERNELS, PEAK, BW)]
    assert got == ["stream", "ridge", "matmul"], f"expected input order, got {got}"


def test_regimes():
    got = [r["regime"] for r in place(KERNELS, PEAK, BW)]
    assert got == ["memory", "compute", "compute"], f"expected ['memory', 'compute', 'compute'], got {got}"


def test_intensity():
    got = [r["intensity"] for r in place(KERNELS, PEAK, BW)]
    assert all(close(a, b) for a, b in zip(got, [1.0, 10.0, 100.0])), f"expected [1, 10, 100], got {got}"


def test_attainable():
    got = [r["attainable"] for r in place(KERNELS, PEAK, BW)]
    want = [10e9, 100e9, 100e9]
    assert all(close(a, b) for a, b in zip(got, want)), f"expected {want}, got {got}"


def test_time_is_the_longer_of_the_two():
    got = [r["time"] for r in place(KERNELS, PEAK, BW)]
    want = [0.1, 0.1, 1.0]
    assert all(close(a, b) for a, b in zip(got, want)), f"expected {want} seconds, got {got}"


def test_time_matches_flops_over_attainable():
    for (name, flops, nbytes), r in zip(KERNELS, place(KERNELS, PEAK, BW)):
        assert close(r["time"], flops / r["attainable"]), f"{name}: time {r['time']} should equal flops/attainable {flops / r['attainable']}"


def test_memory_bound_kernel_cares_about_bytes_not_flops():
    a = place([("k", 1e8, 1e9)], PEAK, BW)[0]
    b = place([("k", 4e8, 1e9)], PEAK, BW)[0]
    assert a["regime"] == b["regime"] == "memory", f"both should be memory-bound, got {a['regime']} and {b['regime']}"
    assert close(a["time"], b["time"]), f"4x the FLOPs at the same traffic should not change a memory-bound time: {a['time']} vs {b['time']}"
