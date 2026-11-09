import numpy as np
from submission import grpo_update, softmax, group_advantages


def objective(theta, theta_old, theta_ref, actions, adv, beta):
    """Equation 3 for one-token outputs; the clip is inactive at theta == theta_old."""
    lp = np.log(softmax(theta))[actions]
    lo = np.log(softmax(theta_old))[actions]
    lr = np.log(softmax(theta_ref))[actions]
    d = lr - lp
    k3 = np.exp(d) - d - 1
    return np.mean(np.exp(lp - lo) * adv - beta * k3)


def numeric_grad(theta, theta_ref, actions, adv, beta, h=1e-6):
    g = np.zeros_like(theta)
    for k in range(len(theta)):
        up, dn = theta.copy(), theta.copy()
        up[k] += h
        dn[k] -= h
        g[k] = (objective(up, theta, theta_ref, actions, adv, beta) - objective(dn, theta, theta_ref, actions, adv, beta)) / (2 * h)
    return g


THETA = np.array([0.3, -0.2, 0.5, 0.0])
REF = np.array([0.0, 0.1, -0.4, 0.2])
ACTIONS = np.array([2, 0, 2, 1, 3, 2, 0, 2])
REWARDS = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])


def test_gradient_matches_finite_differences_without_kl():
    adv = group_advantages(REWARDS)
    want = numeric_grad(THETA, REF, ACTIONS, adv, 0.0)
    got = (grpo_update(THETA, ACTIONS, REWARDS, REF, 1.0, 0.0) - THETA)
    assert np.allclose(got, want, atol=1e-5), f"expected {want}, got {got}"


def test_gradient_matches_finite_differences_with_kl():
    adv = group_advantages(REWARDS)
    for beta in (0.04, 0.5):
        want = numeric_grad(THETA, REF, ACTIONS, adv, beta)
        got = (grpo_update(THETA, ACTIONS, REWARDS, REF, 1.0, beta) - THETA)
        assert np.allclose(got, want, atol=1e-5), f"beta={beta}: expected {want}, got {got}"


def test_learning_rate_scales_the_step():
    a = grpo_update(THETA, ACTIONS, REWARDS, REF, 0.1, 0.04) - THETA
    b = grpo_update(THETA, ACTIONS, REWARDS, REF, 0.3, 0.04) - THETA
    assert np.allclose(3 * a, b, atol=1e-9), f"a step with 3x the learning rate must be 3x as long: expected {3 * a}, got {b}"


def test_good_arm_goes_up_bad_arm_goes_down():
    new = grpo_update(np.zeros(4), ACTIONS, REWARDS, np.zeros(4), 0.5, 0.0)
    p = softmax(new)
    assert p[2] > 0.25, f"arm 2 always earned reward 1, so its probability must rise above 0.25, got {p[2]:.4f}"
    assert p[0] < 0.25 and p[1] < 0.25, f"arms 0 and 1 always earned 0, so they must fall below 0.25, got {p[:2]}"


def test_kl_term_pulls_back_toward_the_reference():
    # All rewards equal, so every advantage is 0 and only the KL term acts.
    theta = np.array([2.0, 0.0, -1.0, 0.5])
    ref = np.zeros(4)
    actions = np.array([0, 0, 0, 1, 2, 3, 0, 1])
    rewards = np.ones(8)

    def kl(t):
        p, q = softmax(t), softmax(ref)
        return float(np.sum(p * np.log(p / q)))

    new = grpo_update(theta, actions, rewards, ref, 0.2, 1.0)
    assert kl(new) < kl(theta), f"KL to the reference must shrink: expected below {kl(theta):.4f}, got {kl(new):.4f}"
    same = grpo_update(theta, actions, rewards, theta.copy(), 0.2, 1.0)
    assert np.allclose(same, theta, atol=1e-9), f"with pi == pi_ref and zero advantages nothing moves; expected {theta}, got {same}"


def test_inputs_are_not_modified():
    th, ref, a, r = THETA.copy(), REF.copy(), ACTIONS.copy(), REWARDS.copy()
    grpo_update(th, a, r, ref, 0.5, 0.04)
    assert np.array_equal(th, THETA), f"logits changed: expected {THETA}, got {th}"
    assert np.array_equal(ref, REF), f"reference logits changed: expected {REF}, got {ref}"


def test_training_loop_finds_the_best_arm():
    p_arm = np.array([0.2, 0.5, 0.8])
    for seed in range(5):
        rng = np.random.default_rng(seed)
        theta, ref = np.zeros(3), np.zeros(3)
        for _ in range(100):
            a = rng.choice(3, size=8, p=softmax(theta))
            r = (rng.random(8) < p_arm[a]).astype(float)
            theta = grpo_update(theta, a, r, ref, 0.5, 0.0)
        p = softmax(theta)[2]
        assert p > 0.9, f"seed {seed}: after 100 steps the best arm should have probability above 0.9, got {p:.3f}"
