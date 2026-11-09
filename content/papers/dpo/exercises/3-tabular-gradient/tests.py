import numpy as np
from submission import dpo_grad, dpo_loss, log_softmax

REF = np.log(np.array([0.70, 0.25, 0.05]))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def numeric_grad(theta, ref, pairs, beta, eps=1e-6):
    g = np.zeros_like(theta)
    for k in range(len(theta)):
        a, b = theta.copy(), theta.copy()
        a[k] += eps
        b[k] -= eps
        g[k] = (dpo_loss(a, ref, pairs, beta) - dpo_loss(b, ref, pairs, beta)) / (2 * eps)
    return g


def test_gradient_at_the_reference():
    got = dpo_grad(REF, REF, [(2, 0)], 0.5)
    assert np.allclose(got, [0.25, 0.0, -0.25], atol=1e-9), f"expected [0.25, 0, -0.25] (-beta * 0.5 * (e_w - e_l)), got {got}"


def test_matches_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(5):
        theta = rng.normal(size=4)
        ref = rng.normal(size=4)
        pairs = [(0, 1), (3, 2), (1, 3)]
        beta = float(rng.uniform(0.1, 2.0))
        want = numeric_grad(theta, ref, pairs, beta)
        got = dpo_grad(theta, ref, pairs, beta)
        assert np.allclose(got, want, atol=1e-6), f"beta={beta:.3f}: expected {want}, got {got}"


def test_descent_direction_raises_winner_over_loser():
    theta = REF.copy()
    g = dpo_grad(theta, REF, [(2, 0)], 0.5)
    new = theta - 1.0 * g
    old_gap = log_softmax(theta)[2] - log_softmax(theta)[0]
    new_gap = log_softmax(new)[2] - log_softmax(new)[0]
    assert new_gap > old_gap, f"a gradient step must raise log pi(y_w) - log pi(y_l): {old_gap:.4f} -> {new_gap:.4f}"
    assert dpo_loss(new, REF, [(2, 0)], 0.5) < dpo_loss(theta, REF, [(2, 0)], 0.5), "a small gradient step must lower the loss"


def test_wrongly_ordered_pairs_get_more_weight():
    wrong = REF + np.array([2.0, 0.0, -2.0])  # policy moved toward the loser (response 0)
    right = REF + np.array([-2.0, 0.0, 2.0])
    g_wrong = dpo_grad(wrong, REF, [(2, 0)], 0.5)
    g_right = dpo_grad(right, REF, [(2, 0)], 0.5)
    assert np.allclose(g_wrong[2], -0.5 * sigmoid(2.0), atol=1e-9), f"expected {-0.5 * sigmoid(2.0):.4f} at the wrong order, got {g_wrong[2]}"
    assert np.allclose(g_right[2], -0.5 * sigmoid(-2.0), atol=1e-9), f"expected {-0.5 * sigmoid(-2.0):.4f} at the right order, got {g_right[2]}"
    assert abs(g_wrong[2]) > 4 * abs(g_right[2]), f"the wrong order must weigh about 7.4x more: {g_wrong[2]} vs {g_right[2]}"


def test_scales_with_beta_when_the_weight_is_fixed():
    # At pi = pi_ref the weight is 0.5 for every beta, so the gradient is linear in beta.
    g1 = dpo_grad(REF, REF, [(1, 2)], 0.2)
    g2 = dpo_grad(REF, REF, [(1, 2)], 0.4)
    assert np.allclose(g2, 2 * g1), f"expected the gradient to double with beta: {g1} -> {g2}"


def test_mean_over_pairs():
    one = dpo_grad(REF, REF, [(2, 0)], 0.5)
    two = dpo_grad(REF, REF, [(2, 0), (2, 0)], 0.5)
    assert np.allclose(one, two), f"the loss is a mean, so repeating a pair changes nothing: {one} vs {two}"
    mixed = dpo_grad(REF, REF, [(2, 0), (1, 0)], 0.5)
    want = 0.5 * (dpo_grad(REF, REF, [(2, 0)], 0.5) + dpo_grad(REF, REF, [(1, 0)], 0.5))
    assert np.allclose(mixed, want), f"expected the average of two gradients {want}, got {mixed}"


def test_gradient_sums_to_zero():
    rng = np.random.default_rng(1)
    theta, ref = rng.normal(size=5), rng.normal(size=5)
    g = dpo_grad(theta, ref, [(0, 4), (2, 1)], 0.7)
    assert abs(g.sum()) < 1e-12, f"adding a constant to all logits changes nothing, so the gradient must sum to 0; sum = {g.sum()}"
