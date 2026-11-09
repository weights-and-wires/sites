import numpy as np
from submission import ppo_epochs, softmax


def make_batch():
    rng = np.random.default_rng(0)
    means = np.array([0.0, 0.5, 1.0])
    actions = rng.choice(3, size=64, p=np.full(3, 1 / 3))
    rewards = means[actions] + 0.3 * rng.normal(size=64)
    return actions, rewards - rewards.mean()


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def test_one_epoch_is_the_vanilla_policy_gradient():
    actions = np.array([0, 1, 1, 2])
    adv = np.array([1.0, -1.0, 0.5, 2.0])
    theta = np.array([0.1, -0.2, 0.3])
    pi = softmax(theta)
    want = theta + 0.1 * np.mean(adv[:, None] * (np.eye(3)[actions] - pi), axis=0)
    got = ppo_epochs(theta, actions, adv, 0.2, 0.1, 1)
    assert np.allclose(got, want), f"one epoch has r=1, so expected {want}, got {got}"


def test_two_epochs_unclipped_by_hand():
    actions = np.array([0, 1, 1, 2])
    adv = np.array([1.0, -1.0, 0.5, 2.0])
    theta0 = np.array([0.1, -0.2, 0.3])
    lr = 0.3
    pi0 = softmax(theta0)
    onehot = np.eye(3)[actions]
    g0 = np.mean(adv[:, None] * (onehot - pi0), axis=0)
    theta1 = theta0 + lr * g0
    pi1 = softmax(theta1)
    r = pi1[actions] / pi0[actions]
    g1 = np.mean((r * adv)[:, None] * (onehot - pi1), axis=0)
    want = theta1 + lr * g1
    got = ppo_epochs(theta0, actions, adv, None, lr, 2)
    assert np.allclose(got, want), f"expected {want}, got {got}; the second epoch must weight each sample by its ratio {r}"


def test_does_not_modify_input():
    theta = np.zeros(3)
    actions, adv = make_batch()
    ppo_epochs(theta, actions, adv, 0.2, 0.5, 10)
    assert np.all(theta == 0), f"theta_old must not change, got {theta}"


def test_clipping_limits_the_policy_change():
    actions, adv = make_batch()
    theta = np.zeros(3)
    pi0 = softmax(theta)
    clipped = softmax(ppo_epochs(theta, actions, adv, 0.2, 0.5, 50))
    free = softmax(ppo_epochs(theta, actions, adv, None, 0.5, 50))
    assert kl(pi0, clipped) < 0.05, f"clipped KL after 50 epochs should stay small (< 0.05), got {kl(pi0, clipped):.4f}"
    assert kl(pi0, free) > 1.0, f"unclipped KL after 50 epochs should be large (> 1.0), got {kl(pi0, free):.4f}"
    assert clipped[2] / pi0[2] < 1.5, f"best-arm ratio with clipping should stay under 1.5, got {clipped[2] / pi0[2]:.3f}"


def test_clipped_update_stays_bounded_for_many_epochs():
    actions, adv = make_batch()
    theta = np.zeros(3)
    pi0 = softmax(theta)
    pi = softmax(ppo_epochs(theta, actions, adv, 0.2, 0.5, 200))
    assert kl(pi0, pi) < 0.1, f"after 200 epochs the clipped KL should stay under 0.1 (the page shows 0.02), got {kl(pi0, pi):.4f}"
    assert pi[2] < 0.5, f"best-arm probability should stay under 0.5, got {pi[2]:.3f}"


def test_still_improves_the_best_arm():
    actions, adv = make_batch()
    pi = softmax(ppo_epochs(np.zeros(3), actions, adv, 0.2, 0.5, 10))
    assert pi[2] > 0.40 and pi[0] < 0.30, f"expected the best arm above 0.40 and the worst below 0.30, got {pi.round(3)}"
