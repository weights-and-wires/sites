import numpy as np
from submission import patchify, unpatchify


def reference(img, P):
    H, W, C = img.shape
    rows = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            rows.append(img[i:i + P, j:j + P, :].reshape(-1))
    return np.array(rows)


def test_shape():
    img = np.zeros((8, 12, 3))
    out = patchify(img, 4)
    assert out.shape == (6, 48), f"expected shape (6, 48) = (HW/P^2, P^2*C), got {out.shape}"


def test_known_values():
    img = np.arange(16, dtype=float).reshape(4, 4, 1)
    want = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], dtype=float)
    got = patchify(img, 2)
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"


def test_matches_loop_on_rectangular_rgb():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(6, 9, 3))
    want = reference(img, 3)
    got = patchify(img, 3)
    assert got.shape == want.shape, f"expected shape {want.shape}, got {got.shape}"
    assert np.allclose(got, want), "patch order or pixel order inside a patch differs from the raster-order loop"


def test_channel_order_inside_patch():
    img = np.zeros((2, 2, 3))
    img[0, 0] = [1, 2, 3]
    img[0, 1] = [4, 5, 6]
    got = patchify(img, 2)[0]
    assert list(got[:6]) == [1, 2, 3, 4, 5, 6], f"first two pixels must be (row, col, channel) ordered, got {got[:6]}"


def test_patch_of_ones_is_isolated():
    img = np.zeros((8, 8, 1))
    img[4:8, 4:8, 0] = 1.0
    got = patchify(img, 4)
    assert got[3].sum() == 16 and got[:3].sum() == 0, f"only patch 3 should hold ones, row sums are {got.sum(1)}"


def test_roundtrip_is_exact():
    rng = np.random.default_rng(1)
    for H, W, C, P in [(8, 8, 1, 2), (6, 9, 3, 3), (16, 8, 3, 4)]:
        img = rng.normal(size=(H, W, C))
        back = unpatchify(patchify(img, P), H, W, C, P)
        assert back.shape == img.shape, f"expected shape {img.shape}, got {back.shape}"
        assert np.array_equal(back, img), f"round trip must be exact for {(H, W, C, P)}"


def test_unpatchify_known_values():
    patches = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], dtype=float)
    want = np.arange(16, dtype=float).reshape(4, 4, 1)
    got = unpatchify(patches, 4, 4, 1, 2)
    assert np.array_equal(got, want), f"expected\n{want[..., 0]}\ngot\n{got[..., 0]}"
