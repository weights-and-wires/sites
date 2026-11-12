import numpy as np
from submission import critic_loss, ddpg_targets

gamma = 0.9
s = np.array([1.0, 0.0, 2.0])
a = np.array([0.5, 0.5, 1.0])
r = np.array([0.0, 1.0, 2.0])
s_next = np.array([2.0, -1.0, 3.0])
done = np.array([0.0, 0.0, 1.0])


def actor_target(S):
    return 0.5 * S


def critic_target(S, A):
    return S - (A - S) ** 2


def test_targets_on_page_example():
    y = np.asarray(ddpg_targets(r, s_next, done, gamma, actor_target, critic_target))
    want = np.array([0.9, -0.125, 2.0])
    assert y.shape == (3,), f"expected shape (3,), got {y.shape}"
    assert np.allclose(y, want), f"expected {want}, got {y}"


def test_terminal_row_ignores_the_target_networks():
    big = lambda S, A: np.full(len(S), 1e6)
    y = np.asarray(ddpg_targets(r, s_next, done, gamma, actor_target, big))
    assert np.isclose(y[2], 2.0), f"a terminal row must give y = r = 2.0, got {y[2]}"
    assert np.isclose(y[0], 0.0 + gamma * 1e6), f"a non-terminal row must bootstrap, got {y[0]}"


def test_critic_target_gets_the_target_actors_action():
    seen = {}

    def spy_actor(S):
        seen["actor_in"] = np.array(S)
        return np.full(len(S), 7.0)

    def spy_critic(S, A):
        seen["critic_s"] = np.array(S)
        seen["critic_a"] = np.array(A)
        return np.zeros(len(S))

    ddpg_targets(r, s_next, done, gamma, spy_actor, spy_critic)
    assert np.allclose(seen["actor_in"], s_next), f"mu' must read s_next {s_next}, got {seen['actor_in']}"
    assert np.allclose(seen["critic_s"], s_next), f"Q' must read s_next {s_next}, got {seen['critic_s']}"
    assert np.allclose(seen["critic_a"], 7.0), f"Q' must get the action from mu', got {seen['critic_a']}"


def test_gamma_scales_only_the_bootstrap():
    y = np.asarray(ddpg_targets(np.array([1.0]), np.array([0.0]), np.array([0.0]), 0.5,
                                lambda S: S, lambda S, A: np.array([4.0])))
    assert np.allclose(y, [3.0]), f"expected 1 + 0.5 * 4 = 3.0, got {y}"


def test_negative_next_values_are_kept():
    y = np.asarray(ddpg_targets(np.array([0.0]), np.array([0.0]), np.array([0.0]), 1.0,
                                lambda S: S, lambda S, A: np.array([-3.0])))
    assert np.allclose(y, [-3.0]), f"expected -3.0, got {y}"


def test_loss_on_page_example():
    y = ddpg_targets(r, s_next, done, gamma, actor_target, critic_target)
    q = 0.2 * s + a
    got = critic_loss(q, y)
    assert np.isclose(got, 0.263542, atol=1e-4), f"expected 0.26354, got {got}"


def test_loss_is_a_mean_of_squares():
    got = critic_loss(np.array([0.0, 0.0]), np.array([2.0, -1.0]))
    assert np.isclose(got, 2.5), f"errors 2 and -1 give (4 + 1) / 2 = 2.5, got {got}"
