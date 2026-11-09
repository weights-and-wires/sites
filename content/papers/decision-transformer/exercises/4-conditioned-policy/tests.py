import numpy as np
from submission import fit_table, action_probs, rollout, step, N_STATES, START, GOAL, R_MIN


def random_walks(n, seed):
    rng = np.random.default_rng(seed)
    S, A, G = [], [], []
    for _ in range(n):
        s, traj, done = START, [], False
        while not done:
            a = int(rng.integers(2))
            s2, r, done = step(s, a)
            traj.append((s, a, r))
            s = s2
        rtg = np.cumsum([r for _, _, r in traj][::-1])[::-1]
        for (s_, a_, _), g in zip(traj, rtg):
            S.append(s_), A.append(a_), G.append(int(g))
    return np.array(S), np.array(A), np.array(G)


S, A, G = random_walks(500, 0)
TABLE = fit_table(S, A, G)


def test_table_shape_and_total_count():
    assert TABLE.shape == (N_STATES, -R_MIN + 1, 2), f"expected shape {(N_STATES, -R_MIN + 1, 2)}, got {TABLE.shape}"
    assert TABLE.sum() == len(S), f"every timestep must be counted once: expected {len(S)}, got {TABLE.sum()}"


def test_counts_use_every_repeat():
    t = fit_table(np.array([3, 3, 3]), np.array([1, 1, 0]), np.array([-3, -3, -3]))
    assert t[3, -3 - R_MIN].tolist() == [1, 2], f"expected counts [1, 2] for (s=3, R=-3), got {t[3, -3 - R_MIN].tolist()}"


def test_return_index_is_not_off_by_one():
    t = fit_table(np.array([3]), np.array([1]), np.array([-3]))
    assert t[3, -3 - R_MIN, 1] == 1, f"the count for R=-3 must sit at index {-3 - R_MIN}; nonzero at {np.argwhere(t).tolist()}"
    t = fit_table(np.array([2]), np.array([0]), np.array([0]))
    assert t[2, -R_MIN, 0] == 1, f"R=0 must sit at the last return index {-R_MIN}; nonzero at {np.argwhere(t).tolist()}"


def test_returns_below_r_min_share_the_lowest_bucket():
    t = fit_table(np.array([3, 3]), np.array([0, 1]), np.array([-100, R_MIN]))
    assert t[3, 0].tolist() == [1, 1], f"returns <= {R_MIN} belong to bucket 0; got {t[3, 0].tolist()}"
    assert t.sum() == 2, f"expected 2 counts, got {t.sum()}"


def test_conditioning_on_the_shortest_return_always_goes_right():
    p = action_probs(TABLE, START, -3)
    assert np.allclose(p, [0.0, 1.0]), f"at state 3 with R=-3 only the path right-right-right fits; expected [0, 1], got {p}"


def test_return_changes_the_policy():
    p3, p9 = action_probs(TABLE, START, -3), action_probs(TABLE, START, -9)
    assert p3[1] > p9[1] + 0.3, f"P(right) must fall as the target gets worse: R=-3 gave {p3[1]:.3f}, R=-9 gave {p9[1]:.3f}"


def test_probabilities_normalize_over_actions():
    for R in (-3, -5, -8):
        p = action_probs(TABLE, START, R)
        assert np.isclose(p.sum(), 1.0), f"probabilities at R={R} must sum to 1, got {p} (sum {p.sum()})"


def test_no_data_falls_back_to_uniform():
    p = action_probs(TABLE, START, -2)
    assert np.allclose(p, [0.5, 0.5]), f"R=-2 is impossible from state 3, so there is no data; expected [0.5, 0.5], got {p}"
    p = action_probs(TABLE, GOAL, -3)
    assert np.allclose(p, [0.5, 0.5]), f"expected [0.5, 0.5] for an unseen state, got {p}"


def test_returns_outside_the_range_are_clipped():
    p = action_probs(TABLE, START, 5)
    assert np.allclose(p, [0.5, 0.5]), f"a positive target clips to 0, which has no data at state 3; expected [0.5, 0.5], got {p}"
    p = action_probs(TABLE, START, -500)
    assert np.isclose(p.sum(), 1.0), f"a very low target must clip to R_MIN and still give probabilities, got {p}"


def test_rollout_reaches_the_goal_in_exactly_the_target_steps():
    rng = np.random.default_rng(1)
    for target in (-3, -5):
        lens = [rollout(TABLE, target, rng)[0] for _ in range(20)]
        assert lens == [-target] * 20, f"conditioned on R={target} every episode should take {-target} steps, got {lens}"
