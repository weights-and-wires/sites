import numpy as np
from submission import indexer_kl


def hand_case():
    # Row 3 uses the two heads of scripts/kl_warmup_example.py. Rows 0 to 2 match
    # their index scores exactly, so they add 0 to the loss.
    H, L = 2, 4
    attn = np.zeros((H, L, L))
    for h in range(H):
        for t in range(3):
            attn[h, t, : t + 1] = 1.0 / (t + 1)
    attn[0, 3] = [0.1, 0.2, 0.6, 0.1]
    attn[1, 3] = [0.3, 0.1, 0.4, 0.2]
    I = np.full((L, L), -np.inf)
    for t in range(3):
        I[t, : t + 1] = 0.0
    I[3] = [1.0, 0.5, 1.5, 0.0]
    return attn, I


def random_case(L=6, H=3, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.tril(np.ones((L, L), dtype=bool))
    logits = np.where(mask[None], rng.normal(size=(H, L, L)), -np.inf)
    attn = np.exp(logits - logits.max(-1, keepdims=True))
    attn /= attn.sum(-1, keepdims=True)
    I = np.where(mask, rng.normal(size=(L, L)), -np.inf)
    return attn, I


def test_dense_hand_value():
    attn, I = hand_case()
    got = indexer_kl(attn, I)
    assert np.isclose(got, 0.0247, atol=1e-4), f"expected about 0.0247, got {got}"


def test_sparse_hand_value():
    attn, I = hand_case()
    got = indexer_kl(attn, I, k=3)
    assert np.isclose(got, 0.0157, atol=1e-4), f"expected about 0.0157, got {got}"


def test_zero_when_indexer_matches_the_target():
    attn, I = random_case(seed=1)
    p = attn.sum(0)
    p = p / p.sum(-1, keepdims=True)
    with np.errstate(divide="ignore"):
        I_match = np.log(p) + 3.7  # softmax ignores a constant shift
    got = indexer_kl(attn, I_match)
    assert abs(got) < 1e-9, f"expected 0 when softmax(I) equals p, got {got}"


def test_loss_is_positive_for_a_wrong_indexer():
    attn, I = random_case(seed=2)
    got = indexer_kl(attn, I)
    assert got > 0.01, f"expected a clearly positive loss, got {got}"


def test_sparse_with_large_k_equals_dense():
    attn, I = random_case(seed=3)
    dense, sparse = indexer_kl(attn, I), indexer_kl(attn, I, k=100)
    assert np.isclose(dense, sparse), f"expected {dense}, got {sparse}"


def reference(attn, I, k):
    p = attn.sum(0)
    p = p / p.sum(-1, keepdims=True)
    total = 0.0
    for t in range(I.shape[0]):
        S = list(range(t + 1)) if k is None or t + 1 <= k else list(np.argsort(-I[t])[:k])
        a = p[t, S] / p[t, S].sum()
        z = np.exp(I[t, S] - I[t, S].max())
        b = z / z.sum()
        total += float((a * np.log(a / b)).sum())
    return total


def test_matches_loop_reference():
    attn, I = random_case(L=7, seed=5)
    for k in (None, 2, 4):
        want, got = reference(attn, I, k), indexer_kl(attn, I, k)
        assert np.isclose(got, want), f"k={k}: expected {want}, got {got}"


def test_sparse_ignores_scores_outside_the_selected_set():
    attn, I = random_case(L=7, seed=6)
    k, t = 3, 6
    S = set(np.argsort(-I[t])[:k].tolist())
    other = next(s for s in range(t + 1) if s not in S)
    I2 = I.copy()
    I2[t, other] -= 5.0  # still outside the top-k
    a, b = indexer_kl(attn, I, k), indexer_kl(attn, I2, k)
    assert np.isclose(a, b), f"expected {a}, got {b}"
