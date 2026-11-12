import numpy as np
from submission import lowconf_step, random_step

M = 9


def logits_for(probs):
    return np.log(np.asarray(probs, dtype=float))


def test_lowconf_keeps_the_most_confident_predictions():
    # Confidences of the argmax at positions 0..3: 0.6, 0.9, 0.5, 0.8. Keep two: positions 1 and 3.
    probs = [[0.6, 0.4], [0.1, 0.9], [0.5, 0.5], [0.8, 0.2]]
    x = np.full(4, M)
    got = lowconf_step(x, logits_for(probs), M, 2)
    want = [M, 1, M, 0]
    assert got.tolist() == want, f"expected {want}, got {got.tolist()}"


def test_confidence_is_a_probability_not_a_raw_logit():
    # Position 0 has the larger raw logit (5) but a flat distribution; position 1 is peaked.
    logits = np.array([[5.0, 5.0, 5.0], [2.0, -5.0, -5.0]])
    got = lowconf_step(np.full(2, M), logits, M, 1)
    assert got.tolist() == [M, 0], f"expected the peaked position 1 to be kept, got {got.tolist()}"


def test_already_unmasked_tokens_are_never_remasked_or_changed():
    # Position 0 is unmasked with token 1; the model is unsure there and prefers token 0.
    probs = [[0.51, 0.49], [0.99, 0.01], [0.9, 0.1]]
    x = np.array([1, M, M])
    got = lowconf_step(x, logits_for(probs), M, 2)
    assert got.tolist() == [1, 0, M], f"expected [1, 0, {M}], got {got.tolist()}"


def test_result_has_exactly_n_un_unmasked_positions():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((10, 6))
    x = np.full(10, M)
    for n in (0, 1, 4, 10):
        got = lowconf_step(x, logits, M, n)
        assert int((got != M).sum()) == n, f"n_un={n}: expected {n} unmasked positions, got {int((got != M).sum())}"


def test_ties_break_toward_the_lower_index():
    got = lowconf_step(np.full(5, M), np.zeros((5, 3)), M, 2)
    assert got.tolist() == [0, 0, M, M, M], f"expected the first two positions kept, got {got.tolist()}"


def test_lowconf_does_not_modify_the_input():
    x = np.array([M, 2, M])
    keep = x.copy()
    lowconf_step(x, np.random.default_rng(1).standard_normal((3, 4)), M, 3)
    assert np.array_equal(x, keep), f"x_t must not change; expected {keep.tolist()}, got {x.tolist()}"


def test_random_step_fills_the_right_number_with_argmax_tokens():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((8, 5))
    x = np.array([3, M, M, M, 1, M, M, M])
    got = random_step(x, logits, M, 5, np.random.default_rng(0))
    assert int((got != M).sum()) == 5, f"expected 5 unmasked positions, got {int((got != M).sum())}"
    assert got[0] == 3 and got[4] == 1, f"already unmasked tokens must stay; got {got.tolist()}"
    new = (got != M) & (x == M)
    assert (got[new] == logits.argmax(axis=-1)[new]).all(), "newly filled positions must hold the greedy prediction"


def test_random_step_is_seeded_and_uniform():
    logits = np.random.default_rng(3).standard_normal((6, 4))
    x = np.full(6, M)
    a = random_step(x, logits, M, 2, np.random.default_rng(7))
    b = random_step(x, logits, M, 2, np.random.default_rng(7))
    assert np.array_equal(a, b), "the same seed must choose the same positions"
    counts = np.zeros(6)
    for seed in range(3000):
        counts += random_step(x, logits, M, 2, np.random.default_rng(seed)) != M
    freq = counts / 3000
    assert np.allclose(freq, 2 / 6, atol=0.04), f"each position must be filled with probability 2/6; got {freq.round(3).tolist()}"


def test_random_step_ignores_confidence():
    # Position 0 is far more confident, but random remasking must not prefer it.
    probs = [[0.999, 0.001], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    x = np.full(4, M)
    hits = sum(random_step(x, logits_for(probs), M, 1, np.random.default_rng(s))[0] != M for s in range(2000))
    assert 0.18 < hits / 2000 < 0.32, f"position 0 must be filled about 1/4 of the time, got {hits / 2000:.3f}"


def test_random_step_does_not_modify_the_input():
    x = np.array([M, 2, M, M])
    keep = x.copy()
    random_step(x, np.random.default_rng(1).standard_normal((4, 3)), M, 3, np.random.default_rng(0))
    assert np.array_equal(x, keep), f"x_t must not change; expected {keep.tolist()}, got {x.tolist()}"
