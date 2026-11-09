import numpy as np
from submission import BEHAVIORS, generate, grpo_step, rule_reward, softmax, train

ZERO = np.zeros(6)


def surrogate(logits, behaviors, adv):
    p = softmax(logits)
    return float(np.mean([a * np.log(p[b]) for b, a in zip(behaviors, adv)]))


def test_one_step_values():
    new = grpo_step(ZERO, [0, 0, 1, 5], [2, 2, 1, 0], 1.0)
    want = [0.4523, -0.0754, 0.0, 0.0, 0.0, -0.3769]
    assert np.allclose(new, want, atol=1e-3), f"expected {want}, got {np.round(new, 4).tolist()}"


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=6)
    behaviors = [0, 3, 3, 1, 0, 5, 2, 0]
    rewards = [2, 1, 1, 1, 2, 0, 1, 2]
    r = np.array(rewards, float)
    adv = (r - r.mean()) / r.std()
    num = np.zeros(6)
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1e-6
        num[j] = (surrogate(logits + e, behaviors, adv) - surrogate(logits - e, behaviors, adv)) / 2e-6
    got = grpo_step(logits, behaviors, rewards, 1.0) - logits
    assert np.allclose(got, num, atol=1e-4), f"expected {np.round(num, 5).tolist()}, got {np.round(got, 5).tolist()}"


def test_equal_rewards_change_nothing():
    new = grpo_step(ZERO, [0, 1, 2, 5], [1, 1, 1, 1], 1.0)
    assert np.allclose(new, ZERO), f"expected no movement, got {new}"
    assert np.all(np.isfinite(new)), f"got non-finite values {new}"


def test_reward_scale_and_shift_do_not_matter():
    a = grpo_step(ZERO, [0, 1, 5, 0], [2, 1, 0, 2], 0.5)
    b = grpo_step(ZERO, [0, 1, 5, 0], [23, 13, 3, 23], 0.5)
    assert np.allclose(a, b, atol=1e-4), f"rewards r and 10r+3 must give the same step: {np.round(a, 4)} vs {np.round(b, 4)}"


def test_direction_and_input_untouched():
    start = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.5])
    keep = start.copy()
    new = grpo_step(start, [0, 5, 5, 1], [2, 0, 0, 1], 0.3)
    assert np.array_equal(start, keep), "grpo_step must not modify its input"
    assert new[0] > start[0], f"the best behavior must gain logit, got {new[0]} vs {start[0]}"
    assert new[5] < start[5], f"the worst behavior must lose logit, got {new[5]} vs {start[5]}"


def test_logit_sum_is_conserved():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=6)
    new = grpo_step(logits, [0, 1, 2, 3, 4, 5, 0, 0], [2, 1, 1, 1, 1, 0, 2, 2], 0.7)
    assert abs(new.sum() - logits.sum()) < 1e-9, f"softmax gradient sums to 0, but the sum moved from {logits.sum()} to {new.sum()}"


def test_group_size_does_not_scale_the_step():
    one = grpo_step(ZERO, [0, 5], [2, 0], 1.0)
    two = grpo_step(ZERO, [0, 5, 0, 5], [2, 0, 2, 0], 1.0)
    assert np.allclose(one, two, atol=1e-6), f"duplicating the group must not change the step: {one} vs {two}"


def test_training_learns_the_task():
    logits, hist = train(seed=0, steps=60, G=16, lr=1.0)
    assert hist[0] < 0.2, f"the uniform policy starts near 1/6, got {hist[0]}"
    assert hist[-1] > 0.9, f"expected P(behavior 0) above 0.9 after 60 steps, got {hist[-1]:.3f}"
    again = train(seed=0, steps=60, G=16, lr=1.0)[1]
    assert np.allclose(hist, again), "training with the same seed must be reproducible"
