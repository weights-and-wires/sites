import numpy as np
from submission import GAMMA, dqn_step, greedy_reaches_goal, init_params, q_values, train_dqn


def make_batch(rng, B=6, hidden=5):
    p = init_params(rng, hidden=hidden)
    p["b1"] = rng.normal(0, 0.5, hidden)
    p["b2"] = rng.normal(0, 0.5, 4)
    S, S2 = rng.random((B, 2)), rng.random((B, 2))
    A = rng.integers(4, size=B)
    R = rng.normal(size=B)
    D = (rng.random(B) < 0.3).astype(float)
    return p, S, A, R, S2, D


def loss_with_fixed_targets(p, S, A, y):
    q = q_values(p, S)
    return np.mean((y - q[np.arange(len(A)), A]) ** 2)


def copy(p):
    return {k: v.copy() for k, v in p.items()}


def test_returns_the_loss_before_the_update():
    rng = np.random.default_rng(0)
    p, S, A, R, S2, D = make_batch(rng)
    y = R + 0.9 * (1 - D) * q_values(p, S2).max(axis=1)
    want = loss_with_fixed_targets(p, S, A, y)
    got = dqn_step(p, S, A, R, S2, D, 0.9, 0.0)
    assert np.isclose(got, want), f"expected the pre-update loss {want:.6f}, got {got}"


def test_zero_learning_rate_changes_nothing():
    rng = np.random.default_rng(1)
    p, S, A, R, S2, D = make_batch(rng)
    before = copy(p)
    dqn_step(p, S, A, R, S2, D, 0.9, 0.0)
    for k in p:
        assert np.array_equal(p[k], before[k]), f"{k} changed with lr = 0"


def test_update_matches_finite_difference_gradient():
    rng = np.random.default_rng(2)
    p, S, A, R, S2, D = make_batch(rng)
    gamma, lr = 0.9, 0.1
    y = R + gamma * (1 - D) * q_values(p, S2).max(axis=1)  # constants, as in Equation 3
    before = copy(p)
    num = {}
    for k in p:
        g = np.zeros_like(p[k])
        for i in np.ndindex(*p[k].shape):
            hi, lo = copy(before), copy(before)
            hi[k][i] += 1e-6
            lo[k][i] -= 1e-6
            g[i] = (loss_with_fixed_targets(hi, S, A, y) - loss_with_fixed_targets(lo, S, A, y)) / 2e-6
        num[k] = g
    dqn_step(p, S, A, R, S2, D, gamma, lr)
    for k in p:
        want = before[k] - lr * num[k]
        assert np.allclose(p[k], want, atol=1e-6), f"{k}: expected an update of -lr * grad = {want.ravel()[:4]}, got {p[k].ravel()[:4]}"


def test_other_actions_outputs_do_not_move():
    rng = np.random.default_rng(3)
    p, S, A, R, S2, D = make_batch(rng, B=1)
    A = np.array([2])
    before = copy(p)
    dqn_step(p, S, A, R, S2, D, 0.9, 0.1)
    for a in (0, 1, 3):
        assert np.array_equal(p["W2"][:, a], before["W2"][:, a]), f"W2 column {a} must not change when only action 2 was taken"
        assert p["b2"][a] == before["b2"][a], f"b2[{a}] must not change when only action 2 was taken"


def test_terminal_transition_ignores_the_next_state():
    rng = np.random.default_rng(4)
    p, S, A, R, S2, D = make_batch(rng, B=3)
    D = np.ones(3)
    p2 = copy(p)
    dqn_step(p, S, A, R, S2, D, 0.9, 0.1)
    dqn_step(p2, S, A, R, S2 + 50.0, D, 0.9, 0.1)
    for k in p:
        assert np.allclose(p[k], p2[k]), f"{k}: a terminal target must not depend on the next state"


def test_target_uses_the_maximum_over_actions():
    rng = np.random.default_rng(5)
    p, S, A, R, S2, D = make_batch(rng, B=4)
    D = np.zeros(4)
    y = R + 0.9 * q_values(p, S2).max(axis=1)
    got = dqn_step(copy(p), S, A, R, S2, D, 0.9, 0.0)
    want = loss_with_fixed_targets(p, S, A, y)
    assert np.isclose(got, want), f"expected the loss {want:.6f} with max-over-actions targets, got {got}"


def test_a_step_reduces_the_loss_on_the_same_batch():
    rng = np.random.default_rng(6)
    p, S, A, R, S2, D = make_batch(rng)
    first = dqn_step(p, S, A, R, S2, D, 0.9, 0.01)
    # The targets move a little with the parameters, so compare with fixed targets.
    y = R + 0.9 * (1 - D) * q_values(p, S2).max(axis=1)
    assert loss_with_fixed_targets(p, S, A, y) < first + 0.05, "one small step should not raise the loss much"


def test_replay_training_learns_the_gridworld():
    wins = [greedy_reaches_goal(train_dqn(seed)) for seed in (0, 1, 4)]
    assert sum(wins) >= 2, f"expected at least 2 of 3 agents to reach the goal greedily, got {wins}"
