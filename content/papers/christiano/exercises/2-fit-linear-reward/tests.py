import numpy as np
from submission import fit_reward, loss_grad

W_TRUE = np.array([1.0, -2.0, 0.5, 0.0, 1.5])


def make(seed, n, k=15):
    rng = np.random.default_rng(seed)
    F1 = rng.normal(size=(n, k, 5)) + rng.normal(size=(n, 1, 5))
    F2 = rng.normal(size=(n, k, 5)) + rng.normal(size=(n, 1, 5))
    mu = ((F1.sum(1) @ W_TRUE) > (F2.sum(1) @ W_TRUE)).astype(float)
    return F1, F2, mu


def loss(w, F1, F2, mu):
    d = (F1.sum(1) - F2.sum(1)) @ w
    ls = lambda x: -np.logaddexp(0, -x)
    return float(np.mean(-(mu * ls(d) + (1 - mu) * ls(-d))))


def test_gradient_shape_and_zero_point():
    F1, F2, mu = make(0, 20, k=4)
    g = loss_grad(np.zeros(5), F1, F2, mu)
    want = ((0.5 - mu)[:, None] * (F1.sum(1) - F2.sum(1))).mean(0)
    assert g.shape == (5,), f"expected shape (5,), got {g.shape}"
    assert np.allclose(g, want), f"at w = 0 expected {want}, got {g}"


def test_gradient_matches_finite_differences():
    F1, F2, mu = make(1, 30, k=6)
    mu = np.clip(mu, 0.1, 0.9)
    rng = np.random.default_rng(2)
    w = rng.normal(size=5) * 0.3
    g = loss_grad(w, F1, F2, mu)
    num = np.zeros(5)
    for j in range(5):
        e = np.zeros(5)
        e[j] = 1e-5
        num[j] = (loss(w + e, F1, F2, mu) - loss(w - e, F1, F2, mu)) / 2e-5
    assert np.allclose(g, num, atol=1e-6), f"expected finite differences {num}, got {g}"


def test_segment_return_is_a_sum_over_time():
    rng = np.random.default_rng(3)
    F1 = rng.normal(size=(10, 8, 5))
    F2 = rng.normal(size=(10, 8, 5))
    mu = np.full(10, 0.5)
    w = rng.normal(size=5)
    short = loss_grad(w, F1[:, :4], F2[:, :4], mu)
    full = loss_grad(w, np.concatenate([F1[:, :4], F1[:, :4]], 1), np.concatenate([F2[:, :4], F2[:, :4]], 1), mu)
    assert not np.allclose(short, full), "doubling the segment length must change the gradient, because the return is a sum, not a mean"


def test_tie_between_identical_segments_has_no_gradient():
    rng = np.random.default_rng(4)
    F = rng.normal(size=(6, 5, 5))
    g = loss_grad(rng.normal(size=5), F, F.copy(), np.full(6, 0.5))
    assert np.allclose(g, 0), f"identical segments with a tie label must give zero gradient, got {g}"


def test_gradient_points_toward_the_preferred_segment():
    F1 = np.zeros((1, 1, 2))
    F2 = np.zeros((1, 1, 2))
    F1[0, 0] = [1.0, 0.0]
    F2[0, 0] = [0.0, 1.0]
    g = loss_grad(np.zeros(2), F1, F2, np.array([1.0]))
    assert g[0] < 0 < g[1], f"descent must raise the weight on segment 1's feature; expected gradient signs (-, +), got {g}"


def test_fit_recovers_the_true_direction():
    F1, F2, mu = make(5, 300)
    w = fit_reward(F1, F2, mu)
    cos = float(w @ W_TRUE / (np.linalg.norm(w) * np.linalg.norm(W_TRUE)))
    assert cos > 0.97, f"expected cosine similarity above 0.97 with the true weights, got {cos:.3f} (w = {w})"


def test_fit_predicts_held_out_preferences():
    F1, F2, mu = make(6, 300)
    w = fit_reward(F1, F2, mu)
    T1, T2, tmu = make(7, 1000)
    pred = (T1.sum(1) @ w) > (T2.sum(1) @ w)
    acc = float(np.mean(pred == (tmu == 1.0)))
    assert acc > 0.95, f"expected held-out preference accuracy above 0.95, got {acc:.3f}"


def test_fit_lowers_the_loss():
    F1, F2, mu = make(8, 100)
    w = fit_reward(F1, F2, mu, steps=100)
    before, after = loss(np.zeros(5), F1, F2, mu), loss(w, F1, F2, mu)
    assert after < before * 0.5, f"expected the loss to fall below half of {before:.3f}, got {after:.3f}"
