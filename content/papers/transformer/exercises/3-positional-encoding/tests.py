import numpy as np
from submission import sinusoidal_pe


def slow(n, d):
    pe = np.zeros((n, d))
    for pos in range(n):
        for i in range(d // 2):
            pe[pos, 2 * i] = np.sin(pos / 10000 ** (2 * i / d))
            pe[pos, 2 * i + 1] = np.cos(pos / 10000 ** (2 * i / d))
    return pe


def test_shape():
    pe = sinusoidal_pe(7, 6)
    assert pe.shape == (7, 6), f"expected (7, 6), got {pe.shape}"


def test_position_zero():
    pe = sinusoidal_pe(4, 4)
    assert np.allclose(pe[0], [0, 1, 0, 1]), f"expected [0, 1, 0, 1], got {pe[0]}"


def test_position_one_by_hand():
    pe = sinusoidal_pe(4, 4)
    want = [0.8415, 0.5403, 0.0100, 1.0000]
    assert np.allclose(pe[1], want, atol=1e-4), f"expected {want}, got {pe[1]}"


def test_even_sin_odd_cos_interleaved():
    pe = sinusoidal_pe(6, 8)
    assert np.isclose(pe[5, 6], np.sin(5 / 10000 ** (6 / 8))), f"pe[5, 6] should be sin(5 / 10000^(6/8)), got {pe[5, 6]}"
    assert np.isclose(pe[5, 7], np.cos(5 / 10000 ** (6 / 8))), f"pe[5, 7] should be cos(5 / 10000^(6/8)), got {pe[5, 7]}"
    assert np.isclose(pe[3, 2], np.sin(3 / 10000 ** (2 / 8))), f"pe[3, 2] should be sin(3 / 10000^(2/8)), got {pe[3, 2]}"
    assert np.isclose(pe[3, 3], np.cos(3 / 10000 ** (2 / 8))), f"pe[3, 3] should be cos(3 / 10000^(2/8)), got {pe[3, 3]}"


def test_matches_slow_reference():
    got, want = sinusoidal_pe(20, 16), slow(20, 16)
    assert np.allclose(got, want), f"max difference {np.abs(got - want).max():.3g} from the loop reference"


def test_bounded():
    pe = sinusoidal_pe(50, 32)
    assert np.all(np.abs(pe) <= 1 + 1e-12), "sine and cosine values must lie in [-1, 1]"


def test_offset_is_a_rotation():
    # Section 3.5: PE(pos + k) is a linear function of PE(pos). For each (sin, cos) pair,
    # the map is a rotation by k * w_i, where w_i = 1 / 10000^(2i/d).
    d, k, pos = 8, 3, 5
    pe = sinusoidal_pe(20, d)
    for i in range(d // 2):
        w = 1 / 10000 ** (2 * i / d)
        s, c = pe[pos, 2 * i], pe[pos, 2 * i + 1]
        want = [s * np.cos(k * w) + c * np.sin(k * w), c * np.cos(k * w) - s * np.sin(k * w)]
        assert np.allclose(pe[pos + k, 2 * i: 2 * i + 2], want), f"pair {i}: expected {want}, got {pe[pos + k, 2 * i: 2 * i + 2]}"
