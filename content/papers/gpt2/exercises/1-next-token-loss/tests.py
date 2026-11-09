import math

import numpy as np
from submission import lm_loss, perplexity


def test_uniform_logits_give_log_vocab():
    tokens = np.array([3, 1, 4, 1, 5])
    got = lm_loss(np.zeros((5, 7)), tokens)
    assert abs(got - math.log(7)) < 1e-9, f"expected ln 7 = {math.log(7):.6f}, got {got}"


def test_hand_example():
    # Two predictions. Row 0 predicts token 1 with p = 0.5; row 1 predicts token 0 with p = 0.25.
    logits = np.log(np.array([[0.25, 0.5, 0.25], [0.25, 0.25, 0.5], [0.3, 0.3, 0.4]]))
    tokens = np.array([2, 1, 0])
    want = (-math.log(0.5) - math.log(0.25)) / 2
    got = lm_loss(logits, tokens)
    assert abs(got - want) < 1e-9, f"expected {want:.6f} (mean over T - 1 = 2 predictions), got {got}"


def test_row_t_predicts_next_token():
    # Row t puts all its mass on tokens[t + 1], so the shifted loss is near 0.
    tokens = np.array([2, 0, 3, 1, 2])
    logits = np.full((5, 4), -20.0)
    for t in range(4):
        logits[t, tokens[t + 1]] = 20.0
    got = lm_loss(logits, tokens)
    assert got < 1e-6, f"a model that predicts every next token should have loss near 0, got {got}"
    # The same logits scored against the unshifted targets are confidently wrong.
    wrong = lm_loss(logits[::-1].copy(), tokens)
    assert wrong > 10, f"shuffled logits should be confidently wrong (loss > 10), got {wrong}"


def test_last_row_is_ignored():
    tokens = np.array([1, 2, 0, 1])
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5))
    b = a.copy()
    b[-1] = rng.standard_normal(5) * 50
    la, lb = lm_loss(a, tokens), lm_loss(b, tokens)
    assert abs(la - lb) < 1e-12, f"the last row has no target, so it can't change the loss: {la} vs {lb}"


def test_first_token_is_never_a_target():
    logits = np.random.default_rng(1).standard_normal((4, 5))
    a = lm_loss(logits, np.array([0, 2, 1, 3]))
    b = lm_loss(logits, np.array([4, 2, 1, 3]))
    assert abs(a - b) < 1e-12, f"changing tokens[0] must not change the loss: {a} vs {b}"


def test_stable_for_large_logits():
    logits = np.array([[1000.0, 1001.0, 999.0], [1000.0, 1000.0, 1000.0], [0.0, 0.0, 0.0]])
    tokens = np.array([0, 1, 2])
    want = (-(1001 - 1001 - math.log(math.exp(-1) + 1 + math.exp(-2))) + math.log(3)) / 2
    got = lm_loss(logits, tokens)
    assert np.isfinite(got), f"loss overflowed: {got}"
    assert abs(got - want) < 1e-9, f"expected {want:.6f}, got {got}"


def test_perplexity_is_exp_of_loss():
    logits = np.random.default_rng(2).standard_normal((6, 5))
    tokens = np.array([0, 3, 1, 4, 2, 2])
    want = math.exp(lm_loss(logits, tokens))
    got = perplexity(logits, tokens)
    assert abs(got - want) < 1e-9, f"expected exp(loss) = {want:.6f}, got {got}"
    assert abs(perplexity(np.zeros((4, 9)), np.array([1, 2, 3, 4])) - 9) < 1e-9, "uniform over 9 tokens has perplexity 9"
