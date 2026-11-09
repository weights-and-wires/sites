import numpy as np
from submission import switch_forward, switch_backward, train


def make_task(seed=2, n=400):
    """Four clusters; each needs its own linear map of the offset from its center."""
    r = np.random.default_rng(seed)
    centers = np.array([[3, 3], [-3, 3], [-3, -3], [3, -3.0]])
    A = r.normal(size=(4, 2, 2))
    c = r.integers(0, 4, n)
    local = r.normal(size=(n, 2))
    x = np.concatenate([centers[c] + local, np.ones((n, 1))], axis=1)  # last column is a bias
    y = np.einsum("nd,ndo->no", local, A[c])
    return x, y, c


def loss_fn(x, target, Wr, W):
    y, _ = switch_forward(x, Wr, W)
    return float(np.sum((y - target) ** 2) / 2)


def small_problem():
    r = np.random.default_rng(7)
    x = r.normal(size=(5, 3))
    Wr = r.normal(size=(3, 4))
    W = r.normal(size=(4, 3, 2))
    target = r.normal(size=(5, 2))
    return x, Wr, W, target


def test_forward_matches_the_definition():
    x, Wr, W, _ = small_problem()
    y, _ = switch_forward(x, Wr, W)
    h = x @ Wr
    p = np.exp(h - h.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    for t in range(len(x)):
        i = int(np.argmax(p[t]))
        want = p[t, i] * (x[t] @ W[i])
        assert np.allclose(y[t], want), f"token {t}: expected {want}, got {y[t]}"
    assert y.shape == (5, 2), f"expected shape (5, 2), got {y.shape}"


def test_expert_gradient_matches_finite_differences():
    x, Wr, W, target = small_problem()
    y, cache = switch_forward(x, Wr, W)
    _, dW = switch_backward(cache, y - target)
    assert dW.shape == W.shape, f"dW must have the shape of W {W.shape}, got {dW.shape}"
    eps = 1e-6
    for pos in [(0, 0, 0), (1, 2, 1), (3, 1, 0), (2, 0, 1)]:
        Wp, Wm = W.copy(), W.copy()
        Wp[pos] += eps
        Wm[pos] -= eps
        num = (loss_fn(x, target, Wr, Wp) - loss_fn(x, target, Wr, Wm)) / (2 * eps)
        assert np.isclose(dW[pos], num, atol=1e-5), f"dW{pos}: expected {num}, got {dW[pos]}"


def test_router_gradient_matches_finite_differences():
    x, Wr, W, target = small_problem()
    y, cache = switch_forward(x, Wr, W)
    dWr, _ = switch_backward(cache, y - target)
    assert dWr.shape == Wr.shape, f"dWr must have the shape of Wr {Wr.shape}, got {dWr.shape}"
    assert np.abs(dWr).max() > 1e-3, "the router must receive a gradient through the gate value, but dWr is all zeros"
    eps = 1e-6
    for pos in [(0, 0), (1, 1), (2, 3), (0, 2), (1, 0)]:
        Wp, Wm = Wr.copy(), Wr.copy()
        Wp[pos] += eps
        Wm[pos] -= eps
        num = (loss_fn(x, target, Wp, W) - loss_fn(x, target, Wm, W)) / (2 * eps)
        assert np.isclose(dWr[pos], num, atol=1e-5), f"dWr{pos}: expected {num}, got {dWr[pos]}"


def test_unchosen_experts_get_no_gradient():
    x = np.array([[1.0, 0.0], [0.5, 0.5]])
    Wr = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # both tokens pick expert 0
    W = np.ones((3, 2, 2))
    y, cache = switch_forward(x, Wr, W)
    _, dW = switch_backward(cache, np.ones_like(y))
    assert np.allclose(dW[1:], 0), f"experts 1 and 2 saw no tokens, so their gradient must be 0, got {dW[1:]}"
    assert np.abs(dW[0]).sum() > 0, "expert 0 saw both tokens and must have a nonzero gradient"


def test_training_beats_a_single_linear_map():
    x, y, c = make_task()
    B = np.linalg.lstsq(x, y, rcond=None)[0]
    dense_mse = float(np.mean((x @ B - y) ** 2))
    Wr, W, mse = train(x, y, N=4)
    assert mse < 0.1 * dense_mse, f"four experts should fit the task far better than one linear map: expected mse < {0.1 * dense_mse:.3f}, got {mse:.3f} (dense {dense_mse:.3f})"
    _, cache = switch_forward(x, Wr, W)
    idx = cache[2]
    counts = np.bincount(idx, minlength=4)
    assert counts.min() > 40, f"every expert should receive tokens; per-expert counts are {counts.tolist()}"
