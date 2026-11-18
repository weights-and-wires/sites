import numpy as np

from submission import robust_z, spike_episodes


def test_z_hand_example():
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 3.0])
    z = robust_z(y, 4)
    # window [1, 2, 3, 4]: median 2.5, deviations [1.5, 0.5, 0.5, 1.5], MAD 1.0
    want = (100.0 - 2.5) / 1.4826
    assert np.allclose(z[:4], 0), f"entries before w must be 0, got {z[:4]}"
    assert np.isclose(z[4], want, atol=1e-3), f"expected z[4] = {want:.3f}, got {z[4]:.3f}"


def test_window_excludes_current_step():
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 3.0])
    z = robust_z(y, 4)
    # window [2, 3, 4, 100]: median 3.5, deviations [1.5, 0.5, 0.5, 96.5], MAD 1.0
    want = (3.0 - 3.5) / 1.4826
    assert np.isclose(z[5], want, atol=1e-3), f"window for t=5 is y[1:5]; expected {want:.3f}, got {z[5]:.3f}"


def test_robust_to_earlier_spike():
    rng = np.random.default_rng(0)
    y = 2.0 + rng.normal(0, 0.01, 400)
    y[200] += 0.3
    y[300] += 0.08
    z = robust_z(y, 200)
    assert z[300] > 5, f"MAD ignores the earlier spike in the window, so z[300] should exceed 5, got {z[300]:.2f}"


def test_episodes_basic():
    z = np.array([0, 6, 3, 1, 1, 1, 0, 7, 1, 1, 1], dtype=float)
    got = spike_episodes(z)
    assert got == [(1, 5), (7, 10)], f"expected [(1, 5), (7, 10)], got {got}"


def test_calm_counter_resets():
    z = np.array([6, 1, 1, 3, 1, 1, 1, 0], dtype=float)
    got = spike_episodes(z)
    assert got == [(0, 6)], f"z=3 at step 3 resets the calm run; expected [(0, 6)], got {got}"


def test_no_new_episode_inside_one():
    z = np.array([6, 9, 1, 1, 1], dtype=float)
    got = spike_episodes(z)
    assert got == [(0, 4)], f"step 1 is inside the open episode; expected [(0, 4)], got {got}"


def test_open_episode_and_threshold():
    z = np.array([0, 5.0, 0, 6, 1], dtype=float)
    got = spike_episodes(z)
    assert got == [(3, 4)], f"z must be strictly above 5 to start, and an open episode ends at the last index; expected [(3, 4)], got {got}"
