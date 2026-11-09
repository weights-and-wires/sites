import numpy as np
from submission import loss_and_grads


def forward(Xi, Xt, Wi, Wt, t):
    """Independent forward pass with a log-sum-exp, used for finite differences."""
    Ie = Xi @ Wi
    Te = Xt @ Wt
    Ie = Ie / np.linalg.norm(Ie, axis=1, keepdims=True)
    Te = Te / np.linalg.norm(Te, axis=1, keepdims=True)
    L = np.exp(t) * Ie @ Te.T

    def lse(x, axis):
        m = x.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)

    d = np.diag(L)
    return ((lse(L, 1) - d).mean() + (lse(L, 0) - d).mean()) / 2


def setup(seed=0, n=5, di=4, dt_=3, de=3):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, di)), rng.normal(size=(n, dt_)),
            rng.normal(size=(di, de)), rng.normal(size=(dt_, de)), 0.7)


def numeric(f, A, eps=1e-6):
    g = np.zeros_like(A)
    for idx in np.ndindex(*A.shape):
        old = A[idx]
        A[idx] = old + eps
        hi = f()
        A[idx] = old - eps
        lo = f()
        A[idx] = old
        g[idx] = (hi - lo) / (2 * eps)
    return g


def test_loss_matches_forward():
    Xi, Xt, Wi, Wt, t = setup()
    loss, *_ = loss_and_grads(Xi, Xt, Wi, Wt, t)
    want = forward(Xi, Xt, Wi, Wt, t)
    assert np.isclose(loss, want), f"expected loss {want}, got {loss}"


def test_gradient_shapes():
    Xi, Xt, Wi, Wt, t = setup()
    _, dWi, dWt, dt = loss_and_grads(Xi, Xt, Wi, Wt, t)
    assert dWi.shape == Wi.shape, f"dWi: expected shape {Wi.shape}, got {dWi.shape}"
    assert dWt.shape == Wt.shape, f"dWt: expected shape {Wt.shape}, got {dWt.shape}"
    assert np.ndim(dt) == 0, f"dt must be a scalar, got shape {np.shape(dt)}"


def test_image_projection_gradient():
    Xi, Xt, Wi, Wt, t = setup(1)
    _, dWi, _, _ = loss_and_grads(Xi, Xt, Wi, Wt, t)
    want = numeric(lambda: forward(Xi, Xt, Wi, Wt, t), Wi)
    assert np.allclose(dWi, want, atol=1e-5), f"dWi differs from finite differences\nexpected {want}\ngot {dWi}"


def test_text_projection_gradient():
    Xi, Xt, Wi, Wt, t = setup(2)
    _, _, dWt, _ = loss_and_grads(Xi, Xt, Wi, Wt, t)
    want = numeric(lambda: forward(Xi, Xt, Wi, Wt, t), Wt)
    assert np.allclose(dWt, want, atol=1e-5), f"dWt differs from finite differences\nexpected {want}\ngot {dWt}"


def test_temperature_gradient():
    Xi, Xt, Wi, Wt, t = setup(3)
    _, _, _, dt = loss_and_grads(Xi, Xt, Wi, Wt, t)
    eps = 1e-6
    want = (forward(Xi, Xt, Wi, Wt, t + eps) - forward(Xi, Xt, Wi, Wt, t - eps)) / (2 * eps)
    assert np.isclose(dt, want, atol=1e-5), f"expected dt {want}, got {dt}"


def test_gradient_ignores_feature_length():
    # Scaling W_i scales I_f, which normalization removes, so the loss is unchanged
    # and the gradient points orthogonally to W_i: <W_i, dW_i> = 0 (Euler's theorem).
    Xi, Xt, Wi, Wt, t = setup(4)
    loss1, dWi, _, _ = loss_and_grads(Xi, Xt, Wi, Wt, t)
    loss2, _, _, _ = loss_and_grads(Xi, Xt, 3.0 * Wi, Wt, t)
    assert np.isclose(loss1, loss2), f"the loss must not depend on the length of I_f: {loss1} vs {loss2}"
    assert abs(np.sum(Wi * dWi)) < 1e-9, f"expected <Wi, dWi> = 0, got {np.sum(Wi * dWi)}; backpropagate through the normalization"


def test_training_aligns_two_views():
    # Two linear views of the same 3-D latent vectors; gradient descent should pair them up.
    rng = np.random.default_rng(5)
    n = 8
    z = rng.normal(size=(n, 3))
    Xi = z @ rng.normal(size=(3, 6)) + 0.05 * rng.normal(size=(n, 6))
    Xt = z @ rng.normal(size=(3, 5)) + 0.05 * rng.normal(size=(n, 5))
    Wi = 0.3 * rng.normal(size=(6, 4))
    Wt = 0.3 * rng.normal(size=(5, 4))
    t = np.log(1 / 0.07)
    start, *_ = loss_and_grads(Xi, Xt, Wi, Wt, t)
    for _ in range(1500):
        loss, dWi, dWt, _ = loss_and_grads(Xi, Xt, Wi, Wt, t)
        Wi = Wi - 1.0 * dWi
        Wt = Wt - 1.0 * dWt
    Ie = Xi @ Wi
    Te = Xt @ Wt
    Ie = Ie / np.linalg.norm(Ie, axis=1, keepdims=True)
    Te = Te / np.linalg.norm(Te, axis=1, keepdims=True)
    sim = Ie @ Te.T
    acc_i = np.mean(sim.argmax(axis=1) == np.arange(n))
    acc_t = np.mean(sim.argmax(axis=0) == np.arange(n))
    assert loss < 0.25 * start, f"the loss should fall below a quarter of its start {start:.3f}, got {loss:.3f}"
    assert acc_i == 1.0 and acc_t == 1.0, f"every image and text must retrieve its pair; image-to-text {acc_i}, text-to-image {acc_t}"
