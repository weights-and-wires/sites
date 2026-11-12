import numpy as np
from submission import position_ids, tree_mask

#   0 --- 2 --- 4
#   |
#   1 --- 3        5
PARENTS = [-1, 0, 0, 1, 2, -1]


def test_position_ids():
    got = position_ids(PARENTS, 3)
    want = [3, 4, 4, 5, 5, 3]
    assert list(got) == want, f"expected {want}, got {list(got)}"


def test_position_ids_chain_matches_causal_positions():
    got = position_ids([-1, 0, 1, 2], 7)
    assert list(got) == [7, 8, 9, 10], f"expected [7, 8, 9, 10], got {list(got)}"


def test_mask_shape_and_dtype():
    m = tree_mask(PARENTS, 3)
    assert m.shape == (6, 9), f"expected shape (6, 9), got {m.shape}"
    assert np.asarray(m).dtype == bool, f"expected dtype bool, got {np.asarray(m).dtype}"


def test_context_is_visible_to_every_node():
    m = tree_mask(PARENTS, 3)
    assert m[:, :3].all(), f"every node must see all 3 context tokens, got\n{m[:, :3].astype(int)}"


def test_tree_block():
    m = tree_mask(PARENTS, 3)[:, 3:].astype(int)
    want = np.array([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    assert np.array_equal(m, want), f"expected\n{want}\ngot\n{m}"


def test_chain_gives_a_causal_mask():
    m = tree_mask([-1, 0, 1, 2], 2)
    want = np.concatenate([np.ones((4, 2), dtype=bool), np.tril(np.ones((4, 4), dtype=bool))], axis=1)
    assert np.array_equal(m, want), f"a chain must equal a causal mask, expected\n{want.astype(int)}\ngot\n{m.astype(int)}"


def softmax_rows(s):
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def test_one_masked_pass_equals_each_path_alone():
    """Causal consistency: node i's output equals attention over [context, its path]."""
    rng = np.random.default_rng(0)
    ctx, d = 3, 4
    n = len(PARENTS)
    q = rng.normal(size=(n, d))
    k = rng.normal(size=(ctx + n, d))
    v = rng.normal(size=(ctx + n, d))
    m = np.asarray(tree_mask(PARENTS, ctx), dtype=bool)
    scores = np.where(m, q @ k.T / np.sqrt(d), -1e9)
    out = softmax_rows(scores) @ v
    for i in range(n):
        path = []
        j = i
        while j >= 0:
            path.append(j)
            j = PARENTS[j]
        cols = list(range(ctx)) + [ctx + j for j in reversed(path)]
        s = q[i] @ k[cols].T / np.sqrt(d)
        w = np.exp(s - s.max())
        want = (w / w.sum()) @ v[cols]
        assert np.allclose(out[i], want), f"node {i}: expected {want}, got {out[i]}"
