import numpy as np
from submission import masked_cross_entropy


def logits_for(probs):
    return np.log(np.asarray(probs, dtype=float))


def test_hand_value():
    # L = 4, V = 2. Positions 1 and 3 are masked; the true-token probabilities are 0.5 and 0.25.
    probs = [[0.9, 0.1], [0.5, 0.5], [0.3, 0.7], [0.25, 0.75]]
    x0 = np.array([0, 0, 1, 0])
    masked = np.array([False, True, False, True])
    got = masked_cross_entropy(logits_for(probs), x0, masked, 0.5)
    want = 6 * np.log(2)
    assert np.isclose(got, want), f"expected (1/0.5)(ln 2 + ln 4) = {want:.4f}, got {got:.4f}"


def test_unmasked_positions_do_not_contribute():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((6, 5))
    x0 = rng.integers(0, 5, size=6)
    masked = np.array([True, False, True, False, False, True])
    base = masked_cross_entropy(logits, x0, masked, 0.5)
    changed = logits.copy()
    changed[~masked] = rng.standard_normal((3, 5)) * 10
    labels = x0.copy()
    labels[~masked] = (labels[~masked] + 1) % 5
    got = masked_cross_entropy(changed, labels, masked, 0.5)
    assert np.isclose(got, base), f"unmasked logits and labels must not matter; expected {base:.6f}, got {got:.6f}"


def test_no_masked_positions_gives_zero():
    logits = np.random.default_rng(1).standard_normal((4, 3))
    got = masked_cross_entropy(logits, np.array([0, 1, 2, 0]), np.zeros(4, dtype=bool), 0.3)
    assert got == 0.0, f"with nothing masked the loss must be 0, got {got}"


def test_weight_is_one_over_t():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((5, 4))
    x0 = rng.integers(0, 4, size=5)
    masked = np.array([True, True, False, True, False])
    a = masked_cross_entropy(logits, x0, masked, 0.5)
    b = masked_cross_entropy(logits, x0, masked, 0.25)
    assert np.isclose(b, 2 * a), f"halving t must double the loss; expected {2 * a:.6f}, got {b:.6f}"
    c = masked_cross_entropy(logits, x0, masked, 1.0)
    assert np.isclose(a, 2 * c), f"t = 0.5 must give twice the t = 1 loss {c:.6f}, got {a:.6f}"


def test_sum_not_mean_over_masked_positions():
    logits = np.zeros((6, 4))  # uniform model: every position costs ln 4
    x0 = np.zeros(6, dtype=int)
    masked = np.array([True, True, True, False, False, False])
    got = masked_cross_entropy(logits, x0, masked, 1.0)
    assert np.isclose(got, 3 * np.log(4)), f"expected 3 ln 4 = {3 * np.log(4):.4f}, got {got:.4f}"


def test_uniform_model_has_constant_expected_loss():
    # A uniform model has NLL L ln V. With the 1/t weight the expected loss equals it for every t.
    L, V = 8, 16
    x0 = np.zeros(L, dtype=int)
    logits = np.zeros((L, V))
    rng = np.random.default_rng(3)
    for t in (0.2, 0.6):
        vals = [masked_cross_entropy(logits, x0, rng.random(L) < t, t) for _ in range(20000)]
        want = L * np.log(V)
        assert abs(np.mean(vals) - want) < 0.25, f"t={t}: expected a mean near L ln V = {want:.3f}, got {np.mean(vals):.3f}"


def test_large_logits_stay_finite():
    logits = np.array([[1000.0, 0.0, -1000.0], [-1000.0, 1000.0, 0.0]])
    got = masked_cross_entropy(logits, np.array([0, 1]), np.array([True, True]), 1.0)
    assert np.isfinite(got) and abs(got) < 1e-6, f"confident correct logits give a loss near 0, got {got}"
    bad = masked_cross_entropy(logits, np.array([2, 1]), np.array([True, False]), 1.0)
    assert np.isclose(bad, 2000.0), f"expected a loss of 2000, got {bad}"
