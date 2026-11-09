import numpy as np
from submission import mean_pool_heads


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def test_hand_example():
    W = np.array([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=float)  # H = 4 heads of width 1
    got = mean_pool_heads(W, 4, 2)
    want = np.array([[2.0, 6.0], [3.0, 7.0]])
    assert got.shape == (2, 2), f"expected shape (2, 2), got {got.shape}"
    assert np.allclose(got, want), f"heads (0, 1) and (2, 3) should average to {want.tolist()}, got {got.tolist()}"


def test_one_group_is_mean_of_all_heads():
    W = np.array([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=float)
    got = mean_pool_heads(W, 4, 1)
    want = np.array([[4.0], [5.0]])
    assert np.allclose(got, want), f"G = 1 should average all four heads to {want.tolist()}, got {got.tolist()}"


def test_groups_equal_heads_is_identity():
    W = np.random.default_rng(0).normal(size=(6, 8))
    got = mean_pool_heads(W, 4, 4)
    assert got.shape == W.shape, f"G = H keeps the shape {W.shape}, got {got.shape}"
    assert np.allclose(got, W), "G = H has one head per group, so the weights must not change"


def test_wide_heads_average_within_blocks():
    # d_k = 3: each head is a block of 3 columns, so averaging must not mix columns within a head.
    W = np.arange(2 * 4 * 3, dtype=float).reshape(2, 12)
    got = mean_pool_heads(W, 4, 2)
    want = np.stack([(W[:, 0:3] + W[:, 3:6]) / 2, (W[:, 6:9] + W[:, 9:12]) / 2], axis=1).reshape(2, 6)
    assert np.allclose(got, want), f"expected {want.tolist()}, got {got.tolist()}"


def test_groups_are_neighbors():
    # Heads are constant blocks 0, 1, 2, 3, so pooling neighbors gives 0.5 and 2.5; interleaved pooling gives 1 and 2.
    W = np.repeat(np.arange(4, dtype=float), 2)[None, :].repeat(3, axis=0)  # (3, 8), d_k = 2
    got = mean_pool_heads(W, 4, 2)
    assert np.allclose(got[0], [0.5, 0.5, 2.5, 2.5]), f"group 0 averages heads 0 and 1, group 1 averages heads 2 and 3; got {got[0].tolist()}"


def test_is_a_mean_not_a_sum():
    W = np.ones((3, 8))
    got = mean_pool_heads(W, 8, 2)
    assert np.allclose(got, 1.0), f"the mean of ones is one; got {np.unique(got).tolist()}"


def test_matching_heads_convert_exactly():
    # If the heads in a group already agree, the converted layer computes the same attention.
    rng = np.random.default_rng(1)
    d_model, H, G, d_k, n = 8, 4, 2, 2, 5
    X = rng.normal(size=(n, d_model))
    base = rng.normal(size=(d_model, G * d_k))
    Wk = np.repeat(base.reshape(d_model, G, 1, d_k), H // G, axis=2).reshape(d_model, H * d_k)
    pooled = mean_pool_heads(Wk, H, G)
    assert np.allclose(pooled, base), "pooling identical heads must return that head"
    q = rng.normal(size=(n, d_k))
    for i in range(H):
        full = softmax(q @ (X @ Wk[:, i * d_k:(i + 1) * d_k]).T)
        g = i // (H // G)
        shared = softmax(q @ (X @ pooled[:, g * d_k:(g + 1) * d_k]).T)
        assert np.allclose(full, shared), f"head {i} attention changed after conversion"
