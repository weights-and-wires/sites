import numpy as np
from submission import index_scores


def toy():
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    Q = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [1.0, -1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )
    w = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0], [0.5, 0.5]])
    return Q, w, K


def test_shape_and_causal_mask():
    Q, w, K = toy()
    got = index_scores(Q, w, K)
    assert got.shape == (4, 4), f"expected shape (4, 4), got {got.shape}"
    upper = np.triu_indices(4, k=1)
    assert np.all(np.isneginf(got[upper])), f"entries with s > t must be -inf, got {got[upper]}"
    assert np.all(np.isfinite(got[np.tril_indices(4)])), "entries with s <= t must be finite"


def test_hand_checked_cell():
    # Query 3, key 2: dots are 1 and 2, weights are 0.5 and 0.5, so I = 1.5.
    Q, w, K = toy()
    got = index_scores(Q, w, K)[3, 2]
    assert np.isclose(got, 1.5), f"expected I[3, 2] = 1.5, got {got}"


def test_relu_clips_each_head():
    # For query 3 and key 3, the dots are -1 and 0, so both heads clip to 0.
    # Without ReLU the score would be 0.5 * (-1) + 0.5 * 0 = -0.5.
    Q, w, K = toy()
    got = index_scores(Q, w, K)[3, 3]
    assert np.isclose(got, 0.0), f"expected I[3, 3] = 0 after ReLU, got {got}"


def test_matches_loop_reference():
    rng = np.random.default_rng(0)
    L, H, d = 7, 3, 4
    Q, w, K = rng.normal(size=(L, H, d)), rng.normal(size=(L, H)), rng.normal(size=(L, d))
    want = np.full((L, L), -np.inf)
    for t in range(L):
        for s in range(t + 1):
            want[t, s] = sum(w[t, j] * max(Q[t, j] @ K[s], 0.0) for j in range(H))
    got = index_scores(Q, w, K)
    assert np.allclose(got, want), f"expected row 6 {want[6]}, got {got[6]}"


def test_negative_weights_are_allowed():
    # The weights w are not constrained to be positive.
    rng = np.random.default_rng(1)
    L, H, d = 5, 2, 3
    Q, K = rng.normal(size=(L, H, d)), rng.normal(size=(L, d))
    w = -np.abs(rng.normal(size=(L, H)))
    got = index_scores(Q, w, K)
    assert np.all(got[np.tril_indices(L)] <= 1e-12), "negative weights on ReLU outputs must give scores <= 0"
