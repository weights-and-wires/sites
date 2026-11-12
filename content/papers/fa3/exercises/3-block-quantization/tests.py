import numpy as np
from submission import quantize_blocks


def test_hand_checked_row():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    got = quantize_blocks(x, 1)
    want = np.array([[1.0, 2.0, 320 * 4 / 448, 4.0]])
    assert np.allclose(got, want, rtol=1e-12), f"expected {want} (3 rounds to 320 * 4 / 448), got {got}"


def test_block_maximum_is_kept():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 6)) * np.repeat([1.0, 50.0], 4)[:, None]
    got = quantize_blocks(x, 4)
    for i in (0, 4):
        want, have = np.abs(x[i:i + 4]).max(), np.abs(got[i:i + 4]).max()
        assert abs(want - have) < 1e-12 * want, f"block at row {i}: expected amax {want}, got {have}"


def test_ragged_last_block_and_shape():
    x = np.random.default_rng(1).normal(size=(10, 3))
    got = quantize_blocks(x, 4)
    assert got.shape == x.shape, f"expected shape {x.shape}, got {got.shape}"
    rel = np.abs(got - x).max() / np.abs(x).max()
    assert rel < 0.07, f"expected error under 7% of the amax, got {rel:.3f}"


def test_blocks_are_independent():
    # Rows of size ~1 sit next to rows of size ~1e6. One scale per tensor flushes the small rows to zero.
    rng = np.random.default_rng(2)
    x = np.vstack([rng.normal(size=(4, 8)), 1e6 * rng.normal(size=(4, 8))])
    small = x[:4]
    per_block = quantize_blocks(x, 4)[:4]
    per_tensor = quantize_blocks(x, 8)[:4]
    err_block = np.linalg.norm(per_block - small) / np.linalg.norm(small)
    err_tensor = np.linalg.norm(per_tensor - small) / np.linalg.norm(small)
    assert err_block < 0.05, f"expected per-block relative error under 0.05, got {err_block:.4f}"
    assert err_tensor > 0.5, f"expected per-tensor to flush the small rows (error above 0.5), got {err_tensor:.4f}"


def test_zero_block_stays_zero():
    x = np.vstack([np.zeros((2, 3)), np.ones((2, 3))])
    got = quantize_blocks(x, 2)
    assert np.all(np.isfinite(got)), f"expected finite output, got {got}"
    assert np.array_equal(got[:2], np.zeros((2, 3))), f"expected zeros in the first block, got {got[:2]}"


def test_scale_equivariance_and_input_unchanged():
    x = np.random.default_rng(3).normal(size=(6, 5))
    before = x.copy()
    a = quantize_blocks(x, 3)
    b = quantize_blocks(7.0 * x, 3)
    assert np.array_equal(x, before), "quantize_blocks modified its input"
    assert np.allclose(b, 7.0 * a, rtol=1e-9), "expected quantizing 7x to give 7 times the result (the scale absorbs it)"
