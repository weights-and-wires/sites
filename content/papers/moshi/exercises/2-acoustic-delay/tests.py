import numpy as np

from submission import apply_delays, latency_ms, undo_delays

A = np.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [21, 22, 23, 24, 25]])


def test_page_example():
    want = np.array([[1, 2, 3, 4, 5], [0, 11, 12, 13, 14], [0, 21, 22, 23, 24]])
    got = apply_delays(A, [0, 1, 1])
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"


def test_delay_two():
    want = np.array([[1, 2, 3, 4, 5], [0, 0, 11, 12, 13], [0, 0, 21, 22, 23]])
    got = apply_delays(A, [0, 2, 2])
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"


def test_staircase_has_no_wraparound():
    got = apply_delays(A, [0, 1, 2])
    want = np.array([[1, 2, 3, 4, 5], [0, 11, 12, 13, 14], [0, 0, 21, 22, 23]])
    assert np.array_equal(got, want), f"the front must be 0, not wrapped codes; expected\n{want}\ngot\n{got}"


def test_zero_delay_is_identity_and_input_unchanged():
    before = A.copy()
    got = apply_delays(A, [0, 0, 0])
    assert np.array_equal(got, A), f"expected the input back, got\n{got}"
    assert np.array_equal(A, before), "apply_delays must not modify its input"


def test_shape_and_dtype_kept():
    got = apply_delays(A, [0, 3, 3])
    assert got.shape == A.shape, f"expected shape {A.shape}, got {got.shape}"
    assert got.dtype == A.dtype, f"expected dtype {A.dtype}, got {got.dtype}"


def test_undo_round_trip():
    delays = [0, 2, 3]
    got = undo_delays(apply_delays(A, delays), delays)
    want = A[:, : 5 - 3]
    assert np.array_equal(got, want), f"expected the first 2 steps of A,\n{want}\ngot\n{got}"


def test_latency_values():
    assert latency_ms([0] * 8) == 80, f"expected 80 ms, got {latency_ms([0] * 8)}"
    assert latency_ms([0] + [1] * 7) == 160, f"expected 160 ms, got {latency_ms([0] + [1] * 7)}"
    assert latency_ms([0] + [2] * 7) == 240, f"expected 240 ms, got {latency_ms([0] + [2] * 7)}"
    assert latency_ms(list(range(8))) == 640, f"expected 640 ms, got {latency_ms(list(range(8)))}"


def test_latency_frame_length():
    assert latency_ms([0, 1], frame_ms=10) == 20, f"expected 20 ms, got {latency_ms([0, 1], frame_ms=10)}"
