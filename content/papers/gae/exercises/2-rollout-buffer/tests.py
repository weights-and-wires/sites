import numpy as np
from submission import gae_rollout

GAMMA, LAM = 0.9, 0.5


def reference_gae(r, v, last, gamma, lam):
    """GAE for one episode by the direct sum in Equation 16, written independently of the solution."""
    T = len(r)
    nv = list(v[1:]) + [last]
    delta = [r[t] + gamma * nv[t] - v[t] for t in range(T)]
    return np.array([sum((gamma * lam) ** l * delta[t + l] for l in range(T - t)) for t in range(T)])


def two_episodes():
    rng = np.random.default_rng(0)
    r1, v1 = rng.normal(size=5), rng.normal(size=5)
    r2, v2 = rng.normal(size=4), rng.normal(size=4)
    return r1, v1, r2, v2


def test_single_terminated_episode_matches_page_example():
    r = [1.0, 0.0, 0.0, 2.0]
    v = [0.5, 0.4, 0.6, 0.3]
    got = gae_rollout(r, v, [0, 0, 0, 1], 123.0, GAMMA, LAM)
    want = np.array([1.01109, 0.33575, 0.435, 1.7])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got} (last_value must be ignored after a done step)"


def test_two_episodes_equal_the_two_episodes_separately():
    r1, v1, r2, v2 = two_episodes()
    dones = [0, 0, 0, 0, 1, 0, 0, 0, 1]
    got = gae_rollout(np.concatenate([r1, r2]), np.concatenate([v1, v2]), dones, 7.0, GAMMA, LAM)
    want = np.concatenate([reference_gae(r1, v1, 0.0, GAMMA, LAM), reference_gae(r2, v2, 0.0, GAMMA, LAM)])
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_cut_off_episode_bootstraps_from_last_value():
    r1, v1, r2, v2 = two_episodes()
    dones = [0, 0, 0, 0, 1, 0, 0, 0, 0]  # the second episode is cut off after four steps
    got = gae_rollout(np.concatenate([r1, r2]), np.concatenate([v1, v2]), dones, 2.5, GAMMA, LAM)
    want = np.concatenate([reference_gae(r1, v1, 0.0, GAMMA, LAM), reference_gae(r2, v2, 2.5, GAMMA, LAM)])
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_first_episode_does_not_see_the_second():
    # Changing anything in episode 2 must leave the advantages of episode 1 unchanged.
    r1, v1, r2, v2 = two_episodes()
    dones = [0, 0, 0, 0, 1, 0, 0, 0, 1]
    a = gae_rollout(np.concatenate([r1, r2]), np.concatenate([v1, v2]), dones, 0.0, GAMMA, LAM)
    b = gae_rollout(np.concatenate([r1, r2 + 100.0]), np.concatenate([v1, v2 - 50.0]), dones, 0.0, GAMMA, LAM)
    assert np.allclose(a[:5], b[:5]), f"episode 1 advantages changed with episode 2 data: {a[:5]} vs {b[:5]}"


def test_one_step_episodes_are_plain_residuals():
    r = np.array([1.0, 2.0, 3.0])
    v = np.array([0.5, 0.5, 0.5])
    got = gae_rollout(r, v, [1, 1, 1], 9.0, GAMMA, 0.95)
    want = r - v
    assert np.allclose(got, want), f"expected {want} (reward minus value, no bootstrap), got {got}"


def test_no_done_matches_single_trajectory_with_bootstrap():
    rng = np.random.default_rng(5)
    r, v = rng.normal(size=10), rng.normal(size=10)
    got = gae_rollout(r, v, np.zeros(10), 1.5, 0.99, 0.95)
    want = reference_gae(r, v, 1.5, 0.99, 0.95)
    assert np.allclose(got, want), f"expected {want}, got {got}"
