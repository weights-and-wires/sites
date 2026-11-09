import numpy as np

from submission import deepseekmoe_layer

# The router logits of the page: one token, 6 routed experts, top-3.
LOGITS = np.array([2.0, 0.5, 1.0, -1.0, 0.0, 1.5])
SM = np.exp(LOGITS) / np.exp(LOGITS).sum()


def scalar_setup():
    """u = [1.0], centroids chosen so that u @ e_i equals LOGITS[i]."""
    u = np.array([[1.0]])
    centroids = LOGITS[:, None]
    shared = [lambda x: np.full_like(x, 100.0)]
    routed = [(lambda i: (lambda x: np.full_like(x, 10.0 * (i + 1))))(i) for i in range(6)]
    return u, centroids, shared, routed


def test_page_example_output():
    u, c, sh, ro = scalar_setup()
    h, idx = deepseekmoe_layer(u, c, sh, ro, 3)
    assert idx[0].tolist() == [0, 5, 2], f"expected experts [0, 5, 2], got {idx[0].tolist()}"
    assert np.allclose(h[0, 0], 125.1025, atol=1e-3), f"expected 125.1025 (gates are not renormalized), got {h[0, 0]}"


def test_gates_are_not_renormalized():
    u, c, sh, ro = scalar_setup()
    h, _ = deepseekmoe_layer(u, c, sh, ro, 3)
    renorm = 1.0 + 100.0 + (SM[[0, 5, 2]] / SM[[0, 5, 2]].sum() * np.array([10.0, 60.0, 30.0])).sum()
    assert not np.allclose(h[0, 0], renorm, atol=1e-2), f"output {h[0, 0]} equals the renormalized value {renorm}; the paper keeps the raw softmax values"


def test_shared_expert_is_always_applied():
    u, c, sh, ro = scalar_setup()
    h_with, _ = deepseekmoe_layer(u, c, sh, ro, 3)
    h_without, _ = deepseekmoe_layer(u, c, [], ro, 3)
    assert np.allclose(h_with - h_without, 100.0), f"the shared expert must add 100 with weight 1, got a difference of {(h_with - h_without)[0]}"


def test_residual_is_added():
    u = np.array([[3.0, -2.0]])
    h, _ = deepseekmoe_layer(u, np.eye(2), [], [lambda x: 0 * x, lambda x: 0 * x], 1)
    assert np.allclose(h, u), f"with zero experts the output must be the residual u = {u}, got {h}"


def test_each_routed_expert_gets_only_its_tokens_and_total_is_T_times_k():
    rng = np.random.default_rng(0)
    T, d, n, k = 40, 6, 12, 3
    u = rng.normal(size=(T, d))
    centroids = rng.normal(size=(n, d))
    seen = [0] * n

    def make(i):
        def f(x):
            seen[i] += x.shape[0]
            return x * (i + 1)
        return f

    h, idx = deepseekmoe_layer(u, centroids, [], [make(i) for i in range(n)], k)
    assert idx.shape == (T, k), f"expected idx shape {(T, k)}, got {idx.shape}"
    assert sum(seen) == T * k, f"expected {T * k} routed expert rows in total, got {sum(seen)}"
    counts = np.bincount(idx.ravel(), minlength=n).tolist()
    assert seen == counts, f"each expert must run only on the tokens that chose it: expected {counts}, got {seen}"


def test_matches_dense_reference():
    rng = np.random.default_rng(1)
    T, d, n, k = 25, 5, 10, 4
    u = rng.normal(size=(T, d))
    centroids = rng.normal(size=(n, d))
    W = rng.normal(size=(n + 2, d, d)) * 0.3
    shared = [(lambda j: (lambda x: np.tanh(x @ W[n + j])))(j) for j in range(2)]
    routed = [(lambda i: (lambda x: np.tanh(x @ W[i])))(i) for i in range(n)]
    h, idx = deepseekmoe_layer(u, centroids, shared, routed, k)
    z = u @ centroids.T
    s = np.exp(z - z.max(1, keepdims=True))
    s /= s.sum(1, keepdims=True)
    want = u.copy()
    for t in range(T):
        for j in range(2):
            want[t] += shared[j](u[t:t + 1])[0]
        for i in np.argsort(-s[t])[:k]:
            want[t] += s[t, i] * routed[i](u[t:t + 1])[0]
    assert np.allclose(h, want), f"expected the per-token reference {want[0]}, got {h[0]}"


def test_softmax_is_over_experts_not_tokens():
    u = np.array([[1.0], [-1.0]])
    c = np.array([[2.0], [0.0], [-2.0]])
    ro = [(lambda i: (lambda x: np.ones_like(x) * (i + 1)))(i) for i in range(3)]
    h, idx = deepseekmoe_layer(u, c, [], ro, 1)
    e0 = np.exp([2.0, 0.0, -2.0]); e0 /= e0.sum()
    e1 = np.exp([-2.0, 0.0, 2.0]); e1 /= e1.sum()
    want = np.array([[1.0 + e0[0] * 1], [-1.0 + e1[2] * 3]])
    assert np.allclose(h, want), f"expected {want.ravel()}, got {h.ravel()}"
