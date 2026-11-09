import numpy as np

from submission import contrastive_loss, sample_distractors


def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def one_step_loss(c_t, cands, kappa):
    logits = np.array([cos(c_t, x) for x in cands]) / kappa
    m = logits.max()
    return float(m + np.log(np.exp(logits - m).sum()) - logits[0])


C = np.array([[2.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
Q = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 0.5]])


def test_hand_example_kappa_1():
    got = contrastive_loss(C, Q, np.array([0]), np.array([[1, 2]]), 1.0)
    assert abs(got - 0.5561) < 1e-3, f"expected 0.5561 (cos sims 0.9487, 0.3162, -0.6), got {got}"


def test_hand_example_kappa_0_1_is_sharper():
    got = contrastive_loss(C, Q, np.array([0]), np.array([[1, 2]]), 0.1)
    assert abs(got - 0.0018) < 1e-4, f"expected 0.0018 at kappa = 0.1, got {got}"


def test_mean_over_masked_steps():
    idx = np.array([0, 1, 2])
    dis = np.array([[1, 2], [0, 2], [0, 1]])
    want = np.mean([one_step_loss(C[t], [Q[t]] + [Q[j] for j in dis[i]], 0.5) for i, t in enumerate(idx)])
    got = contrastive_loss(C, Q, idx, dis, 0.5)
    assert abs(got - want) < 1e-9, f"expected the mean of the per-step losses {want}, got {got}"


def test_uses_cosine_not_dot_product():
    big = Q.copy()
    big[0] *= 50.0
    a = contrastive_loss(C, Q, np.array([0]), np.array([[1, 2]]), 1.0)
    b = contrastive_loss(C, big, np.array([0]), np.array([[1, 2]]), 1.0)
    assert abs(a - b) < 1e-9, f"cosine ignores the norm of q: expected {a}, got {b} after scaling q_t by 50"
    c2 = C.copy()
    c2[0] *= 7.0
    d = contrastive_loss(c2, Q, np.array([0]), np.array([[1, 2]]), 1.0)
    assert abs(a - d) < 1e-9, f"cosine ignores the norm of c: expected {a}, got {d} after scaling c_t by 7"


def test_stable_for_small_kappa():
    got = contrastive_loss(C * 1e3, Q, np.array([0]), np.array([[1, 2]]), 1e-3)
    assert np.isfinite(got), f"loss must stay finite for tiny kappa, got {got}"
    want = one_step_loss(C[0], [Q[0], Q[1], Q[2]], 1e-3)
    assert abs(got - want) < 1e-6 * max(1.0, want), f"expected {want}, got {got}"


def test_chance_level_with_identical_candidates():
    c = np.ones((6, 4))
    q = np.ones((6, 4))
    idx = np.arange(6)
    dis = np.array([[j for j in range(6) if j != t][:3] for t in idx])
    got = contrastive_loss(c, q, idx, dis, 0.1)
    assert abs(got - np.log(4)) < 1e-9, f"K = 3 equal candidates: expected log(4) = {np.log(4)}, got {got}"


def test_only_masked_steps_are_scored():
    # Step 1 has a terrible target; leaving it out of masked_idx must remove its loss.
    c = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    q = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    good = contrastive_loss(c, q, np.array([0]), np.array([[3, 2]]), 1.0)
    both = contrastive_loss(c, q, np.array([0, 1]), np.array([[3, 2], [0, 2]]), 1.0)
    assert both > good + 0.5, f"adding a badly predicted masked step must raise the mean: {good} then {both}"


def test_distractors_shape_masked_only_never_self():
    rng = np.random.default_rng(0)
    idx = np.array([3, 4, 5, 9, 10, 11, 12, 20])
    d = sample_distractors(idx, 4, rng)
    assert d.shape == (8, 4), f"shape: expected (8, 4), got {d.shape}"
    for i, t in enumerate(idx):
        assert t not in d[i], f"row {i}: the positive step {t} must not be its own distractor, got {d[i].tolist()}"
        assert set(d[i].tolist()) <= set(idx.tolist()), f"row {i}: distractors must be masked steps, got {d[i].tolist()}"
        assert len(set(d[i].tolist())) == 4, f"row {i}: distractors must be distinct, got {d[i].tolist()}"


def test_distractors_are_roughly_uniform():
    rng = np.random.default_rng(1)
    idx = np.arange(10)
    counts = np.zeros(10)
    for _ in range(400):
        d = sample_distractors(idx, 3, rng)
        for v in d[0]:
            counts[v] += 1
    assert counts[0] == 0, f"step 0 is the positive of row 0: expected 0 draws, got {counts[0]}"
    freq = counts[1:] / counts[1:].sum()
    assert np.all(np.abs(freq - 1 / 9) < 0.03), f"expected about {1 / 9:.3f} each, got {np.round(freq, 3).tolist()}"


def test_too_few_masked_steps_raises():
    try:
        sample_distractors(np.array([1, 2, 3]), 5, np.random.default_rng(0))
    except ValueError:
        return
    assert False, "expected ValueError when K exceeds the number of other masked steps"
