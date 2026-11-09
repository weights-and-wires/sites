import numpy as np
from submission import embed_patches


def make(seed=0, H=6, W=9, C=3, P=3, D=5):
    rng = np.random.default_rng(seed)
    N = (H // P) * (W // P)
    img = rng.normal(size=(H, W, C))
    E = rng.normal(size=(P * P * C, D))
    cls = rng.normal(size=(D,))
    pos = rng.normal(size=(N + 1, D))
    return img, P, E, cls, pos, N


def patch(img, P, k):
    gw = img.shape[1] // P
    i, j = divmod(k, gw)
    return img[i * P:(i + 1) * P, j * P:(j + 1) * P, :].reshape(-1)


def test_shape():
    img, P, E, cls, pos, N = make()
    z = embed_patches(img, P, E, cls, pos)
    assert z.shape == (N + 1, 5), f"expected shape {(N + 1, 5)} = (N+1, D), got {z.shape}"


def test_class_token_is_row_zero_with_position():
    img, P, E, cls, pos, N = make()
    z = embed_patches(img, P, E, cls, pos)
    assert np.allclose(z[0], cls + pos[0]), f"row 0 expected x_class + E_pos[0] = {cls + pos[0]}, got {z[0]}"


def test_patch_rows_use_the_right_position():
    img, P, E, cls, pos, N = make(seed=3)
    z = embed_patches(img, P, E, cls, pos)
    for k in range(N):
        want = patch(img, P, k) @ E + pos[k + 1]
        assert np.allclose(z[k + 1], want), f"row {k + 1} expected patch {k} @ E + E_pos[{k + 1}] = {want}, got {z[k + 1]}"


def test_zero_projection_leaves_class_and_positions():
    img, P, E, cls, pos, N = make()
    z = embed_patches(img, P, np.zeros_like(E), cls, pos)
    want = np.vstack([cls[None, :], np.zeros((N, E.shape[1]))]) + pos
    assert np.allclose(z, want), f"with E = 0 expected the class token plus E_pos, got\n{z}"


def test_no_bias_and_linear_in_the_image():
    img, P, E, cls, pos, N = make()
    z1 = embed_patches(img, P, E, cls, pos)
    z2 = embed_patches(2 * img, P, E, cls, pos)
    base = np.vstack([cls[None, :], np.zeros((N, E.shape[1]))]) + pos
    assert np.allclose(z2 - base, 2 * (z1 - base)), "patch embeddings must be linear in the pixels (no bias term)"


def test_does_not_modify_inputs():
    img, P, E, cls, pos, N = make()
    pos0, cls0 = pos.copy(), cls.copy()
    embed_patches(img, P, E, cls, pos)
    assert np.array_equal(pos, pos0) and np.array_equal(cls, cls0), "inputs must not be changed in place"
