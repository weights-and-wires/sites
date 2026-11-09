import numpy as np
from submission import tp_mlp_backward, gelu, dgelu

rng = np.random.default_rng(1)
X = rng.normal(size=(4, 6))
A = rng.normal(size=(6, 12))
B = rng.normal(size=(12, 6))
dZ = rng.normal(size=(4, 6))


def reference():
    H = X @ A
    Y = gelu(H)
    dY = dZ @ B.T
    dH = dY * dgelu(H)
    return dH @ A.T, X.T @ dH, Y.T @ dZ


def test_dgelu_matches_finite_difference():
    x = np.linspace(-3, 3, 13)
    eps = 1e-6
    want = (gelu(x + eps) - gelu(x - eps)) / (2 * eps)
    assert np.allclose(dgelu(x), want, atol=1e-6), f"expected {want}, got {dgelu(x)}"


def test_dx_matches_single_device():
    dX_ref, _, _ = reference()
    for n in (1, 2, 3, 6):
        dX, _, _ = tp_mlp_backward(X, A, B, dZ, n)
        assert np.allclose(dX, dX_ref), f"n={n}: expected dX {dX_ref[0]}, got {dX[0]}"


def test_dx_matches_finite_difference():
    # L = sum(Z * dZ), so dL/dX = dX.
    dX, _, _ = tp_mlp_backward(X, A, B, dZ, 3)
    eps = 1e-6
    i, j = 2, 4
    Xp, Xm = X.copy(), X.copy()
    Xp[i, j] += eps
    Xm[i, j] -= eps
    L = lambda Xv: np.sum((gelu(Xv @ A) @ B) * dZ)
    want = (L(Xp) - L(Xm)) / (2 * eps)
    assert abs(dX[i, j] - want) < 1e-5, f"expected dX[{i},{j}] = {want}, got {dX[i, j]}"


def test_weight_gradient_blocks():
    _, dA_ref, dB_ref = reference()
    _, dA_blocks, dB_blocks = tp_mlp_backward(X, A, B, dZ, 4)
    assert len(dA_blocks) == 4 and len(dB_blocks) == 4, f"expected 4 blocks each, got {len(dA_blocks)} and {len(dB_blocks)}"
    assert all(b.shape == (6, 3) for b in dA_blocks), f"expected dA blocks of shape (6, 3), got {[b.shape for b in dA_blocks]}"
    assert all(b.shape == (3, 6) for b in dB_blocks), f"expected dB blocks of shape (3, 6), got {[b.shape for b in dB_blocks]}"
    assert np.allclose(np.hstack(dA_blocks), dA_ref), "the column blocks of dA must concatenate to the single-device dA"
    assert np.allclose(np.vstack(dB_blocks), dB_ref), "the row blocks of dB must concatenate to the single-device dB"


def test_one_gpu_alone_is_not_enough():
    # The local gradient on GPU 0 covers only its own columns of A.
    dX_ref, _, _ = reference()
    H = X @ np.split(A, 4, axis=1)[0]
    local = (dZ @ np.split(B, 4, axis=0)[0].T * dgelu(H)) @ np.split(A, 4, axis=1)[0].T
    assert not np.allclose(local, dX_ref), "a single GPU's local dX should differ from the full dX"
    dX, _, _ = tp_mlp_backward(X, A, B, dZ, 4)
    assert np.allclose(dX, dX_ref), f"after the all-reduce, expected {dX_ref[0]}, got {dX[0]}"
