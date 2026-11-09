import numpy as np
from submission import sample, sample_probs

LOGITS = np.array([2.0, 1.0, 0.5, -1.0])


def test_plain_softmax():
    want = np.array([0.6095, 0.2242, 0.1360, 0.0303])
    got = sample_probs(LOGITS)
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_temperature_divides_logits():
    want = np.array([0.8420, 0.1140, 0.0419, 0.0021])
    got = sample_probs(LOGITS, temperature=0.5)
    assert np.allclose(got, want, atol=1e-4), f"temperature 0.5 should sharpen the distribution; expected {want}, got {got}"
    hot = sample_probs(LOGITS, temperature=2.0)
    assert hot[0] < 0.45 and hot[3] > 0.09, f"temperature 2 should flatten the distribution; got {hot}"


def test_top_k_two():
    want = np.array([0.7311, 0.2689, 0.0, 0.0])
    got = sample_probs(LOGITS, top_k=2)
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_top_k_one_is_greedy():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(20)
    p = sample_probs(z, top_k=1)
    want = np.zeros(20)
    want[np.argmax(z)] = 1.0
    assert np.allclose(p, want), f"top_k = 1 must put all mass on the argmax {np.argmax(z)}, got {p}"


def test_sums_to_one_and_keeps_largest():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(30) * 3
    for k in (1, 3, 10):
        p = sample_probs(z, temperature=0.7, top_k=k)
        assert abs(p.sum() - 1) < 1e-9, f"k = {k}: probabilities must sum to 1, got {p.sum()}"
        assert set(np.flatnonzero(p)) == set(np.argsort(z)[-k:]), f"k = {k}: wrong tokens kept: {np.flatnonzero(p)}"


def test_temperature_and_top_k_together():
    want = np.array([0.8808, 0.1192, 0.0, 0.0])
    got = sample_probs(LOGITS, temperature=0.5, top_k=2)
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_large_k_changes_nothing():
    a = sample_probs(LOGITS, top_k=100)
    b = sample_probs(LOGITS)
    assert np.allclose(a, b), f"top_k above the vocabulary size must keep everything; expected {b}, got {a}"


def test_stable_for_large_logits():
    p = sample_probs(np.array([1000.0, 999.0, 0.0]), top_k=2)
    assert np.all(np.isfinite(p)) and abs(p.sum() - 1) < 1e-9, f"expected finite probabilities that sum to 1, got {p}"


def test_sample_never_draws_a_removed_token():
    rng = np.random.default_rng(0)
    draws = {sample(LOGITS, rng, top_k=2) for _ in range(300)}
    assert draws == {0, 1}, f"with top_k = 2 only tokens 0 and 1 can appear, got {sorted(draws)}"
