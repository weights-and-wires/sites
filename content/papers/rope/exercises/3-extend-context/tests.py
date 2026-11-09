import numpy as np
from submission import rope_angles, interpolate_positions, ntk_base


def test_angles_by_hand():
    # d = 8 gives theta = [1, 0.1, 0.01, 0.001].
    got = rope_angles(np.array([0, 2, 10]), 8)
    want = np.array([[0, 0, 0, 0], [2, 0.2, 0.02, 0.002], [10, 1, 0.1, 0.01]])
    assert got.shape == (3, 4), f"expected shape (3, 4), got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_angles_custom_base():
    got = rope_angles(np.array([5]), 4, base=100.0)
    assert np.allclose(got, [[5.0, 0.5]]), f"base=100, d=4 gives theta = [1, 0.1]; expected [[5, 0.5]], got {got}"


def test_interpolation_shrinks_positions():
    got = interpolate_positions(np.array([0, 4096, 8191]), 2048, 8192)
    want = [0.0, 1024.0, 2047.75]
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_interpolation_keeps_angles_in_trained_range():
    d = 8
    trained = rope_angles(np.array([2048]), d).max()  # the trained length, 2048, bounds the new angles
    extended = rope_angles(interpolate_positions(np.arange(8192), 2048, 8192), d).max()
    assert extended <= trained + 1e-9, f"largest angle {extended} exceeds the trained maximum {trained}"


def test_interpolation_identity_when_not_extending():
    pos = np.array([0, 5, 17])
    got = interpolate_positions(pos, 2048, 2048)
    assert np.allclose(got, pos), f"new_len == train_len should change nothing; expected {pos}, got {got}"


def test_ntk_base_value():
    got = ntk_base(10000.0, 8, 4.0)
    want = 63496.04207872797
    assert np.isclose(got, want), f"expected {want}, got {got}"


def test_ntk_keeps_fastest_pair_and_divides_slowest_by_s():
    d, s = 128, 8.0
    old = rope_angles(np.array([1]), d)[0]
    new = rope_angles(np.array([1]), d, base=ntk_base(10000.0, d, s))[0]
    assert np.isclose(new[0], old[0]), f"pair 0 must keep theta = 1; got {new[0]} against {old[0]}"
    assert np.isclose(new[-1], old[-1] / s), f"last pair should shrink by s = {s}; expected {old[-1] / s}, got {new[-1]}"


def test_ntk_scales_middle_pairs_less_than_interpolation():
    d, s = 128, 4.0
    old = rope_angles(np.array([1]), d)[0]
    new = rope_angles(np.array([1]), d, base=ntk_base(10000.0, d, s))[0]
    ratio = new / old
    assert np.all(np.diff(ratio) < 0), "the ratio should fall steadily from 1 to 1/s across the pairs"
    assert ratio[d // 4] > 1 / s + 0.1, f"the middle pair should shrink by less than s; ratio {ratio[d // 4]}"
