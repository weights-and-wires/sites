import numpy as np

from submission import paged_attention


def dense(q, K, V):
    s = K @ q / np.sqrt(q.shape[-1])
    e = np.exp(s - s.max())
    return (e / e.sum()) @ V


def build(n, B, d, table, num_blocks, seed=0, garbage=50.0):
    """Scatter a sequence of n tokens over the physical blocks in `table`; fill the rest with garbage."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=d)
    K, V = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    pool_K = np.full((num_blocks, B, d), garbage)
    pool_V = np.full((num_blocks, B, d), -garbage)
    for tok in range(n):
        pool_K[table[tok // B], tok % B] = K[tok]
        pool_V[table[tok // B], tok % B] = V[tok]
    return q, K, V, pool_K, pool_V


def check(n, B, d, table, num_blocks, seed=0):
    q, K, V, pK, pV = build(n, B, d, table, num_blocks, seed)
    got = paged_attention(q, pK, pV, table, n)
    want = dense(q, K, V)
    assert got.shape == (d,), f"expected shape ({d},), got {got.shape}"
    assert np.allclose(got, want, atol=1e-8), f"n={n}, B={B}, table={table}: expected {want[:3]}..., got {got[:3]}..."


def test_full_blocks_in_order():
    check(8, 4, 5, [0, 1], 2)


def test_scattered_blocks():
    check(12, 4, 6, [5, 0, 3], 6)


def test_partly_filled_last_block():
    check(10, 4, 8, [4, 0, 5], 6)


def test_single_token():
    check(1, 4, 3, [2], 3)
    q, K, V, pK, pV = build(1, 4, 3, [2], 3)
    assert np.allclose(paged_attention(q, pK, pV, [2], 1), V[0]), "one key gets weight 1, so the output is its value"


def test_unfilled_slots_are_ignored():
    q, K, V, pK, pV = build(5, 4, 4, [1, 0], 2, seed=3)
    want = dense(q, K, V)
    pK2, pV2 = pK.copy(), pV.copy()
    pK2[0, 1:], pV2[0, 1:] = 1e3, 1e3     # change only the 3 unfilled slots of block 0
    got = paged_attention(q, pK2, pV2, [1, 0], 5)
    assert np.allclose(got, want, atol=1e-8), f"unfilled slots changed the output: expected {want}, got {got}"


def test_softmax_spans_all_blocks():
    # Block 0 has tiny scores, block 1 has huge ones. A per-block softmax would weight them equally.
    d = 2
    q = np.array([1.0, 0.0])
    pK = np.zeros((2, 2, d))
    pV = np.zeros((2, 2, d))
    pK[0] = [[0.0, 0.0], [0.0, 0.0]]
    pK[1] = [[10.0, 0.0], [10.0, 0.0]]
    pV[0] = [[1.0, 0.0], [1.0, 0.0]]
    pV[1] = [[0.0, 1.0], [0.0, 1.0]]
    got = paged_attention(q, pK, pV, [0, 1], 4)
    want = dense(q, pK.reshape(4, d), pV.reshape(4, d))
    assert np.allclose(got, want, atol=1e-8), f"expected {want}, got {got}; the softmax must normalize across blocks"
    assert got[1] > 0.99, f"block 1 has far larger scores, so it should dominate; got {got}"


def test_table_is_followed_not_block_index():
    # Logical blocks 0 and 1 live in physical blocks 2 and 0; physical block 1 holds garbage.
    q, K, V, pK, pV = build(8, 4, 4, [2, 0], 3, seed=7)
    got = paged_attention(q, pK, pV, [2, 0], 8)
    assert np.allclose(got, dense(q, K, V), atol=1e-8), f"expected {dense(q, K, V)}, got {got}; read blocks through the table"


def test_permuting_physical_blocks_changes_nothing():
    q, K, V, pK, pV = build(11, 4, 4, [2, 0, 1], 3, seed=9)
    perm = [2, 0, 1]                   # new position of old block i is perm[i]
    pK2, pV2 = np.empty_like(pK), np.empty_like(pV)
    for old, new in enumerate(perm):
        pK2[new], pV2[new] = pK[old], pV[old]
    a = paged_attention(q, pK, pV, [2, 0, 1], 11)
    b = paged_attention(q, pK2, pV2, [perm[2], perm[0], perm[1]], 11)
    assert np.allclose(a, b, atol=1e-10), f"the answer must not depend on physical placement: {a} vs {b}"


def test_scaling_and_stability():
    d = 16
    q, K, V, pK, pV = build(9, 4, d, [0, 1, 2], 3, seed=11)
    got = paged_attention(q * 20000, pK, pV, [0, 1, 2], 9)
    want = dense(q * 20000, K, V)
    assert np.all(np.isfinite(got)), "large scores overflowed exp; subtract the maximum first"
    assert np.allclose(got, want, atol=1e-6), f"expected {want[:3]}..., got {got[:3]}..."
    one = paged_attention(q, pK, pV, [0, 1, 2], 9)
    assert np.allclose(one, dense(q, K, V), atol=1e-8), "the 1/sqrt(d) scale is part of the score"
