import math
from submission import forward_grid, backward_grid, waves, sm_utilization


def test_fa1_grid_ignores_sequence_length():
    a = forward_grid(2, 16, 512, 128, False)
    b = forward_grid(2, 16, 16384, 128, False)
    assert a == 32 and b == 32, f"expected 32 blocks for both lengths, got {a} and {b}"


def test_fa2_grid_adds_row_blocks():
    got = forward_grid(1, 16, 16384, 128, True)
    assert got == 16 * 128, f"expected {16 * 128}, got {got}"


def test_row_blocks_round_up():
    got = forward_grid(1, 1, 1000, 128, True)  # ceil(1000/128) = 8
    assert got == 8, f"expected 8 row blocks for N=1000, Br=128, got {got}"


def test_backward_grid_uses_column_blocks():
    got = backward_grid(2, 8, 4096, 64)
    assert got == 2 * 8 * 64, f"expected {2 * 8 * 64}, got {got}"
    got = backward_grid(1, 1, 1000, 128)
    assert got == 8, f"expected 8 for N=1000, Bc=128, got {got}"


def test_waves():
    assert waves(108) == 1, f"108 blocks fit on 108 SMs in one wave, got {waves(108)}"
    assert waves(109) == 2, f"109 blocks need two waves, got {waves(109)}"
    assert waves(32) == 1, f"expected 1, got {waves(32)}"
    assert waves(100, sms=10) == 10, f"expected 10, got {waves(100, sms=10)}"


def test_utilization_small_grid():
    got = sm_utilization(32)
    assert math.isclose(got, 32 / 108), f"32 blocks on 108 SMs: expected {32 / 108:.4f}, got {got:.4f}"


def test_utilization_tail_wave():
    got = sm_utilization(109)
    assert math.isclose(got, 109 / 216), f"109 blocks take 2 waves of 108: expected {109 / 216:.4f}, got {got:.4f}"
    got = sm_utilization(216)
    assert math.isclose(got, 1.0), f"216 blocks fill 2 waves: expected 1.0, got {got}"


def test_fa2_beats_fa1_on_long_sequences_with_small_batch():
    fa1 = sm_utilization(forward_grid(1, 16, 16384, 128, False))
    fa2 = sm_utilization(forward_grid(1, 16, 16384, 128, True))
    assert fa2 > 0.9 > fa1, f"expected FA2 above 0.9 and FA1 below, got fa1={fa1:.3f}, fa2={fa2:.3f}"
