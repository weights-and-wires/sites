import numpy as np
from submission import block_backward, flash_attention_backward


def forward(Q, K, V):
    S = Q @ K.T
    m = S.max(axis=1)
    E = np.exp(S - m[:, None])
    l = E.sum(axis=1)
    return (E / l[:, None]) @ V, m, l


def naive_backward(Q, K, V, dO):
    S = Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    dP = dO @ V.T
    dS = P * (dP - (P * dP).sum(axis=1, keepdims=True))
    return dS @ K, dS.T @ Q, P.T @ dO


def make(N=10, M=9, d=4, seed=0):
    rng = np.random.default_rng(seed)
    Q, K, V, dO = rng.normal(size=(N, d)), rng.normal(size=(M, d)), rng.normal(size=(M, d)), rng.normal(size=(N, d))
    O, m, l = forward(Q, K, V)
    return Q, K, V, O, dO, m, l


def check(got, want, label):
    for name, g, w in zip(("dQ", "dK", "dV"), got, want):
        assert g.shape == w.shape, f"{label}: {name} expected shape {w.shape}, got {g.shape}"
        assert np.allclose(g, w, atol=1e-9), f"{label}: {name} max error {np.abs(g - w).max():.2e}\nexpected\n{w}\ngot\n{g}"


def test_single_block_equals_naive():
    Q, K, V, O, dO, m, l = make()
    got = block_backward(Q, K, V, O, dO, m, l)
    check(got, naive_backward(Q, K, V, dO), "one block")


def test_blocked_equals_naive():
    Q, K, V, O, dO, m, l = make()
    got = flash_attention_backward(Q, K, V, O, dO, m, l, 3, 4)
    check(got, naive_backward(Q, K, V, dO), "Br=3, Bc=4")


def test_square_blocks_and_odd_sizes():
    Q, K, V, O, dO, m, l = make(N=7, M=7, seed=3)
    got = flash_attention_backward(Q, K, V, O, dO, m, l, 2, 2)
    check(got, naive_backward(Q, K, V, dO), "Br=Bc=2")


def test_uses_global_statistics_not_block_local_ones():
    # One column block holds only part of each row, so P must use the full-row m and l.
    Q, K, V, O, dO, m, l = make(N=4, M=6, seed=5)
    dq, dk, dv = block_backward(Q, K[:3], V[:3], O, dO, m, l)
    S = Q @ K[:3].T
    P = np.exp(S - m[:, None]) / l[:, None]
    assert np.allclose(dv, P.T @ dO, atol=1e-9), f"dV part expected\n{P.T @ dO}\ngot\n{dv}"
    assert P.sum(axis=1).max() < 1.0, "test setup: the block weights must sum to less than 1"


def test_equal_keys_give_zero_dQ():
    # Equal keys give equal scores, so every row of P is uniform and dS sums
    # to zero across keys. Then dQ = dS @ K is 0 when all key rows are equal.
    Q, K, V, _, dO, _, _ = make(N=5, M=8, seed=2)
    ones = np.ones_like(K)
    O1, m1, l1 = forward(Q, ones, V)
    dQ, _, _ = flash_attention_backward(Q, ones, V, O1, dO, m1, l1, 2, 3)
    assert np.allclose(dQ, 0, atol=1e-9), f"all keys equal: dQ must be 0, got max {np.abs(dQ).max():.2e}"


def test_matches_finite_differences():
    Q, K, V, O, dO, m, l = make(N=4, M=5, d=3, seed=7)
    dQ, dK, dV = flash_attention_backward(Q, K, V, O, dO, m, l, 2, 2)

    def loss(Q, K, V):
        return float((forward(Q, K, V)[0] * dO).sum())

    eps = 1e-6
    for arr, grad, name in ((Q, dQ, "dQ"), (K, dK, "dK"), (V, dV, "dV")):
        fd = np.zeros_like(arr)
        for a in range(arr.shape[0]):
            for b in range(arr.shape[1]):
                p, q = arr.copy(), arr.copy()
                p[a, b] += eps
                q[a, b] -= eps
                args_p = [p if x is arr else x for x in (Q, K, V)]
                args_q = [q if x is arr else x for x in (Q, K, V)]
                fd[a, b] = (loss(*args_p) - loss(*args_q)) / (2 * eps)
        assert np.allclose(grad, fd, atol=1e-5), f"{name}: max difference from finite differences {np.abs(grad - fd).max():.2e}"
