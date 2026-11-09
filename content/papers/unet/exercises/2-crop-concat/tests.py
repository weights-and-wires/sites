import numpy as np
from submission import crop_and_concat


def test_shape():
    out = crop_and_concat(np.zeros((3, 10, 10)), np.zeros((5, 6, 6)))
    assert out.shape == (8, 6, 6), f"expected shape (8, 6, 6), got {out.shape}"


def test_paper_level_zero_shape():
    out = crop_and_concat(np.zeros((64, 568, 568)), np.zeros((64, 392, 392)))
    assert out.shape == (128, 392, 392), f"expected (128, 392, 392), got {out.shape}"


def test_center_crop_values():
    skip = np.arange(36, dtype=float).reshape(1, 6, 6)
    up = np.full((1, 4, 4), -1.0)
    out = crop_and_concat(skip, up)
    want = np.array([[7, 8, 9, 10], [13, 14, 15, 16], [19, 20, 21, 22], [25, 26, 27, 28]], dtype=float)
    assert np.array_equal(out[0], want), f"expected the center crop\n{want}\ngot\n{out[0]}"


def test_up_channels_are_second_and_unchanged():
    skip = np.ones((2, 6, 6))
    up = np.arange(16, dtype=float).reshape(1, 4, 4)
    out = crop_and_concat(skip, up)
    assert np.array_equal(out[2], up[0]), f"channel 2 must be the up-convolution output, got\n{out[2]}"
    assert np.all(out[:2] == 1), f"channels 0 and 1 must be the cropped skip, got {out[:2].ravel()}"


def test_channels_stay_separate():
    skip = np.stack([np.full((6, 6), 1.0), np.full((6, 6), 2.0)])
    up = np.zeros((1, 2, 2))
    out = crop_and_concat(skip, up)
    assert np.all(out[0] == 1) and np.all(out[1] == 2), f"each skip channel must keep its own values, got {out[:2].mean(axis=(1, 2))}"


def test_asymmetric_crop():
    skip = np.arange(25, dtype=float).reshape(1, 5, 5)
    up = np.zeros((1, 2, 2))
    out = crop_and_concat(skip, up)
    want = np.array([[6, 7], [11, 12]], dtype=float)
    assert np.array_equal(out[0], want), f"odd difference starts the crop at (H - h) // 2 = 1; expected\n{want}\ngot\n{out[0]}"


def test_same_size_is_plain_concat():
    skip = np.arange(8, dtype=float).reshape(2, 2, 2)
    up = np.arange(4, dtype=float).reshape(1, 2, 2)
    out = crop_and_concat(skip, up)
    assert np.array_equal(out[:2], skip), f"with equal sizes the skip is unchanged, got\n{out[:2]}"
