import numpy as np
from submission import patchify, unpatchify


def test_shape_for_paper_setup():
    z = np.zeros((32, 32, 4))
    for p, T in ((8, 16), (4, 64), (2, 256)):
        got = patchify(z, p).shape
        assert got == (T, p * p * 4), f"p={p}: expected {(T, p * p * 4)}, got {got}"


def test_tiny_example_from_page():
    z = np.arange(16).reshape(4, 4, 1)
    got = patchify(z, 2)
    want = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"


def test_patches_do_not_mix():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(8, 8, 3))
    tok = patchify(z, 4)
    # token 1 is the top-right 4x4 block (rows 0-3, columns 4-7).
    want = z[0:4, 4:8, :].reshape(-1)
    assert np.allclose(tok[1], want), f"token 1 should be the top-right block\nexpected {want[:6]}...\ngot {tok[1][:6]}..."
    # token 2 is the bottom-left block (rows 4-7, columns 0-3).
    want = z[4:8, 0:4, :].reshape(-1)
    assert np.allclose(tok[2], want), f"token 2 should be the bottom-left block\nexpected {want[:6]}...\ngot {tok[2][:6]}..."


def test_channels_are_fastest_axis():
    z = np.zeros((2, 2, 3))
    z[0, 0] = [1, 2, 3]
    tok = patchify(z, 1)
    assert np.array_equal(tok[0], [1, 2, 3]), f"with p=1 token 0 is the pixel's channels [1, 2, 3], got {tok[0]}"
    assert tok.shape == (4, 3), f"expected shape (4, 3), got {tok.shape}"


def test_round_trip():
    rng = np.random.default_rng(1)
    for I, p, C in ((8, 2, 4), (12, 4, 1), (16, 8, 3)):
        z = rng.normal(size=(I, I, C))
        back = unpatchify(patchify(z, p), p, C)
        assert back.shape == z.shape, f"I={I}, p={p}: expected shape {z.shape}, got {back.shape}"
        assert np.allclose(back, z), f"I={I}, p={p}: round trip must return the input"


def test_unpatchify_layout():
    tok = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
    got = unpatchify(tok, 2, 1)[:, :, 0]
    want = np.arange(16).reshape(4, 4)
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"
