import numpy as np
from submission import make_grid, quant, rtn

W = np.array([[-1.0, 0.2, 0.4, 2.0],
              [0.1, 0.4, 0.9, 0.3]])


def test_grid_parameters():
    scale, zero, maxq = make_grid(W, 2)
    assert maxq == 3, f"expected maxq 3 for 2 bits, got {maxq}"
    assert np.allclose(scale, [1.0, 0.3]), f"expected scale [1.0, 0.3], got {scale}"
    assert np.allclose(zero, [1.0, 0.0]), f"expected zero [1.0, 0.0], got {zero}"


def test_grid_includes_zero_for_positive_rows():
    # Row 1 has only positive values. The grid still has to start at 0, not at 0.1.
    scale, zero, _ = make_grid(W, 2)
    assert zero[1] == 0, f"expected zero point 0 for an all-positive row, got {zero[1]}"
    assert np.isclose(scale[1], 0.9 / 3), f"expected scale 0.3 (range 0 to 0.9), got {scale[1]}"


def test_quant_values():
    got = rtn(W, 2)
    want = np.array([[-1.0, 0.0, 0.0, 2.0], [0.0, 0.3, 0.9, 0.3]])
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_rounds_to_nearest_not_floor():
    scale, zero, maxq = np.array([0.25]), np.array([0.0]), 15
    got = quant(np.array([[0.37, 0.39, 0.12]]), scale[:, None], zero[:, None], maxq)
    want = np.array([[0.25, 0.5, 0.0]])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_clips_to_the_code_range():
    scale, zero, maxq = np.array([0.25]), np.array([0.0]), 3
    got = quant(np.array([[5.0, -5.0]]), scale[:, None], zero[:, None], maxq)
    want = np.array([[0.75, 0.0]])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_error_at_most_half_a_step_and_levels():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 40))
    for bits in (2, 3, 4):
        scale, zero, maxq = make_grid(A, bits)
        Q = rtn(A, bits)
        err = np.abs(A - Q).max(axis=1)
        assert np.all(err <= scale / 2 + 1e-12), f"{bits} bits: error {err} exceeds half a step {scale / 2}"
        for r in range(5):
            assert len(np.unique(np.round(Q[r], 9))) <= 2**bits, f"{bits} bits: row {r} has more than {2**bits} levels"


def test_zero_is_representable():
    A = np.array([[-0.7, 0.1, 0.9, 0.35]])
    scale, zero, maxq = make_grid(A, 3)
    got = quant(np.array([[0.0]]), scale[:, None], zero[:, None], maxq)
    assert np.allclose(got, 0.0), f"expected 0.0 to quantize to exactly 0.0, got {got}"


def test_zero_point_is_an_integer_code():
    A = np.array([[-0.4, 0.3, 1.0]])
    scale, zero, maxq = make_grid(A, 2)
    assert np.allclose(zero, [1.0]), f"expected the zero point round(0.4 / (1.4 / 3)) = 1, got {zero}"
    got = quant(np.array([[0.0]]), scale[:, None], zero[:, None], maxq)
    assert np.allclose(got, 0.0), f"expected 0.0 to quantize to exactly 0.0, got {got}"
