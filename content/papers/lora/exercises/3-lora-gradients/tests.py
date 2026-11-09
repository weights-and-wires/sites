import numpy as np
from submission import lora_forward, lora_grads

rng = np.random.default_rng(0)
n, d, k, r = 6, 4, 5, 3
X = rng.normal(size=(n, k))
W0 = rng.normal(size=(d, k))
A = rng.normal(size=(r, k))
B = rng.normal(size=(d, r))
G = rng.normal(size=(n, d))
alpha = 6.0


def loss(A_, B_):
    return float(np.sum(lora_forward(X, W0, A_, B_, alpha) * G))


def numeric(param, which, eps=1e-6):
    out = np.zeros_like(param)
    for idx in np.ndindex(*param.shape):
        p = param.copy()
        p[idx] += eps
        m = param.copy()
        m[idx] -= eps
        if which == "A":
            out[idx] = (loss(p, B) - loss(m, B)) / (2 * eps)
        else:
            out[idx] = (loss(A, p) - loss(A, m)) / (2 * eps)
    return out


def test_shapes():
    dA, dB = lora_grads(X, A, B, alpha, G)
    assert dA.shape == A.shape, f"dA expected shape {A.shape}, got {dA.shape}"
    assert dB.shape == B.shape, f"dB expected shape {B.shape}, got {dB.shape}"


def test_page_example():
    x = np.array([[1.0, 2.0, 3.0]])
    A1 = np.array([[1.0, 0.0, -1.0]])
    B1 = np.array([[1.0], [2.0]])
    dA, dB = lora_grads(x, A1, B1, 2.0, np.array([[1.0, 1.0]]))
    assert np.allclose(dB, [[-4.0], [-4.0]]), f"dB expected [[-4], [-4]], got {dB}"
    assert np.allclose(dA, [[6.0, 12.0, 18.0]]), f"dA expected [[6, 12, 18]], got {dA}"


def test_db_matches_finite_differences():
    _, dB = lora_grads(X, A, B, alpha, G)
    want = numeric(B, "B")
    assert np.allclose(dB, want, atol=1e-5), f"dB max error {np.abs(dB - want).max():.3g}\nexpected\n{want}\ngot\n{dB}"


def test_da_matches_finite_differences():
    dA, _ = lora_grads(X, A, B, alpha, G)
    want = numeric(A, "A")
    assert np.allclose(dA, want, atol=1e-5), f"dA max error {np.abs(dA - want).max():.3g}\nexpected\n{want}\ngot\n{dA}"


def test_scale_enters_both_gradients():
    dA1, dB1 = lora_grads(X, A, B, 3.0, G)
    dA2, dB2 = lora_grads(X, A, B, 6.0, G)
    assert np.allclose(dB2, 2 * dB1), "dB must be linear in alpha"
    assert np.allclose(dA2, 2 * dA1), "dA must be linear in alpha"


def test_zero_b_blocks_the_gradient_of_a():
    B0 = np.zeros((d, r))
    dA, dB = lora_grads(X, A, B0, alpha, G)
    assert np.all(dA == 0), f"with B = 0, dA must be exactly 0, got max |dA| = {np.abs(dA).max()}"
    assert np.abs(dB).max() > 0, "with B = 0, dB must still be nonzero, or training can't start"
