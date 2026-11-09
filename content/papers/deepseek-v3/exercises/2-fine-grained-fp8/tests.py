import numpy as np

from submission import quantize, per_tensor_roundtrip, tile_scales, tile_roundtrip, block_roundtrip


def make_matrix():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (256, 512))
    x[3, 7] = 100000.0
    return x


def rel_rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(x**2)))


def test_tile_scales_values_and_shape():
    x = np.zeros((2, 8))
    x[0, :4] = [1.0, -4.0, 2.0, 0.5]
    x[0, 4:] = [0.1, 0.2, 0.0, -0.05]
    x[1, :4] = [448.0, 1.0, 1.0, 1.0]
    s = tile_scales(x, 4)
    want = np.array([[448 / 4.0, 448 / 0.2], [1.0, 1.0]])
    assert s.shape == (2, 2), f"expected shape (2, 2), got {s.shape}"
    assert np.allclose(s, want), f"expected {want}, got {s} (use |x|, and 1.0 for an all-zero tile)"


def test_tiles_run_along_rows():
    x = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    s = tile_scales(x, 2)
    want = np.array([[448 / 2, 448 / 4], [448 / 6, 448 / 8], [448 / 10, 448 / 12], [448 / 14, 448 / 16]])
    assert np.allclose(s, want), f"a 1 x 2 tile has two neighbors in one row: expected {want}, got {s}"


def test_tile_roundtrip_shape_and_exact_on_representable():
    x = np.array([[1.0, 2.0, 4.0, 8.0], [0.5, 0.25, 1.0, 2.0]])
    y = tile_roundtrip(x, 2)
    assert y.shape == x.shape, f"expected shape {x.shape}, got {y.shape}"
    assert np.allclose(y, x), f"powers of two are exact in E4M3 after any scale: expected {x}, got {y}"


def test_tile_roundtrip_unscales():
    x = np.full((1, 4), 3.0)
    y = tile_roundtrip(x, 4)
    assert np.allclose(y, 3.0, rtol=0.07), f"output must be back on the input's scale: expected about 3.0, got {y}"


def test_tile_roundtrip_negative_and_zero():
    x = np.array([[-7.0, 0.0, 3.0, -0.5], [0.0, 0.0, 0.0, 0.0]])
    y = tile_roundtrip(x, 4)
    assert np.all(y[1] == 0.0), f"an all-zero tile stays zero, got {y[1]}"
    assert y[0, 0] < 0 and y[0, 3] < 0 and y[0, 1] == 0.0, f"signs and zeros must survive, got {y[0]}"


def test_tile_roundtrip_rescues_outlier_neighbors():
    x = make_matrix()
    fine = rel_rmse(x[100:], tile_roundtrip(x, 128)[100:])
    coarse = rel_rmse(x[100:], per_tensor_roundtrip(x)[100:])
    assert fine < 0.035, f"expected rows 100+ rel RMSE below 0.035 with 1x128 tiles, got {fine:.4f}"
    assert coarse > 3 * fine, f"per-tensor error {coarse:.4f} should be over 3x the tile error {fine:.4f}"


def test_outlier_only_hurts_its_own_tile():
    x = make_matrix()
    y = tile_roundtrip(x, 128)
    other = rel_rmse(x[3, 128:], y[3, 128:])
    assert other < 0.035, f"the neighbor tile in row 3 must not feel the outlier, got rel RMSE {other:.4f}"
    other_rows = rel_rmse(x[4:, :128], y[4:, :128])
    assert other_rows < 0.035, f"rows below the outlier must not feel it, got rel RMSE {other_rows:.4f}"


def test_block_roundtrip_outlier_spreads_down_the_block():
    x = make_matrix()
    y = block_roundtrip(x, 128)
    assert y.shape == x.shape, f"expected shape {x.shape}, got {y.shape}"
    inside = rel_rmse(x[4:128, :128], y[4:128, :128])
    outside = rel_rmse(x[128:, :128], y[128:, :128])
    assert inside > 0.08, f"in the outlier's 128x128 block, the shared scale hurts every row: expected rel RMSE above 0.08, got {inside:.4f}"
    assert outside < 0.035, f"the block below must be unaffected: expected rel RMSE below 0.035, got {outside:.4f}"


def test_block_between_tile_and_tensor():
    x = make_matrix()
    e_tile = rel_rmse(x[100:], tile_roundtrip(x, 128)[100:])
    e_block = rel_rmse(x[100:], block_roundtrip(x, 128)[100:])
    e_tensor = rel_rmse(x[100:], per_tensor_roundtrip(x)[100:])
    assert e_tile < e_block < e_tensor, f"expected tile < block < tensor, got {e_tile:.4f}, {e_block:.4f}, {e_tensor:.4f}"
