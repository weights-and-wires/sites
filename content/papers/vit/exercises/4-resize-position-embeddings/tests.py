import numpy as np
from submission import resize_pos_embed


def ramp(g, a=2.0, b=3.0, c=1.0):
    r, s = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    return (a * r + b * s + c).reshape(-1, 1)


def with_cls(grid_rows, cls_value=-7.0):
    return np.vstack([np.full((1, grid_rows.shape[1]), cls_value), grid_rows])


def test_shape():
    out = resize_pos_embed(np.zeros((1 + 4, 6)), 3)
    assert out.shape == (10, 6), f"expected shape (10, 6) = (1 + 3*3, D), got {out.shape}"


def test_class_row_is_kept():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(1 + 9, 4))
    out = resize_pos_embed(E, 5)
    assert np.array_equal(out[0], E[0]), f"row 0 must be copied unchanged: expected {E[0]}, got {out[0]}"


def test_same_size_is_identity():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(1 + 16, 3))
    out = resize_pos_embed(E, 4)
    assert np.allclose(out, E), f"resizing 4x4 to 4x4 must change nothing; max diff {np.abs(out - E).max():.3g}"


def test_2x2_to_3x3_known_values():
    E = with_cls(np.array([[0.0], [1.0], [2.0], [3.0]]))
    want = np.array([[0, 0.5, 1], [1, 1.5, 2], [2, 2.5, 3]]).reshape(-1, 1)
    got = resize_pos_embed(E, 3)[1:]
    assert np.allclose(got, want), f"expected {want.ravel()}, got {got.ravel()}"


def test_linear_ramp_stays_linear():
    g, g_new = 3, 5
    E = with_cls(ramp(g))
    got = resize_pos_embed(E, g_new)[1:].reshape(g_new, g_new)
    c = np.linspace(0, g - 1, g_new)
    want = 2 * c[:, None] + 3 * c[None, :] + 1
    assert np.allclose(got, want), f"a linear ramp must be reproduced exactly; expected\n{want}\ngot\n{got}"


def test_corners_are_preserved_when_upsampling_and_downsampling():
    rng = np.random.default_rng(2)
    for g, g_new in [(4, 9), (9, 4), (14, 24)]:
        E = rng.normal(size=(1 + g * g, 2))
        got = resize_pos_embed(E, g_new)[1:].reshape(g_new, g_new, 2)
        grid = E[1:].reshape(g, g, 2)
        for (i, j), (a, b) in {(0, 0): (0, 0), (0, g_new - 1): (0, g - 1),
                               (g_new - 1, 0): (g - 1, 0), (g_new - 1, g_new - 1): (g - 1, g - 1)}.items():
            assert np.allclose(got[i, j], grid[a, b]), f"{g}->{g_new}: corner {(i, j)} expected {grid[a, b]}, got {got[i, j]}"


def test_rows_and_columns_are_not_swapped():
    g = 3
    E = with_cls(ramp(g, a=1.0, b=0.0, c=0.0))  # value = row index
    got = resize_pos_embed(E, 5)[1:].reshape(5, 5)
    want = np.repeat(np.linspace(0, 2, 5)[:, None], 5, axis=1)
    assert np.allclose(got, want), f"value should grow down the rows only; expected\n{want}\ngot\n{got}"


def test_channels_are_independent():
    g = 3
    E = np.hstack([with_cls(ramp(g)), with_cls(ramp(g, a=0.0, b=5.0, c=0.0))])
    got = resize_pos_embed(E, 5)[1:]
    c = np.linspace(0, 2, 5)
    want0 = (2 * c[:, None] + 3 * c[None, :] + 1).reshape(-1)
    want1 = (5 * np.broadcast_to(c[None, :], (5, 5))).reshape(-1)
    assert np.allclose(got[:, 0], want0) and np.allclose(got[:, 1], want1), "each channel must be resized on its own"
