import numpy as np
from submission import sample_smoothed_action, smoothed_action


def test_page_example():
    got = np.asarray(smoothed_action(np.array([0.8, 0.8, 0.8]), np.array([0.5, 3.0, -4.0]), 0.2, 0.5, -1.0, 1.0))
    want = np.array([0.9, 1.0, 0.3])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_noise_is_clipped_after_scaling():
    got = np.asarray(smoothed_action(np.array([0.0]), np.array([3.0]), 0.2, 0.5, -10.0, 10.0))
    assert np.allclose(got, [0.5]), f"sigma * z = 0.6 must clip to c = 0.5, got {got}"


def test_noise_clip_applies_before_the_bounds():
    got = np.asarray(smoothed_action(np.array([0.0]), np.array([100.0]), 1.0, 0.5, -1.0, 1.0))
    assert np.allclose(got, [0.5]), f"noise clips to 0.5 so the action is 0.5, not the bound 1.0; got {got}"


def test_negative_side():
    got = np.asarray(smoothed_action(np.array([0.0]), np.array([-100.0]), 1.0, 0.5, -1.0, 1.0))
    assert np.allclose(got, [-0.5]), f"expected -0.5, got {got}"


def test_action_bounds_are_enforced():
    got = np.asarray(smoothed_action(np.array([0.9, -0.9]), np.array([2.0, -2.0]), 0.2, 0.5, -1.0, 1.0))
    assert np.allclose(got, [1.0, -1.0]), f"0.9 + 0.4 and -0.9 - 0.4 must clip to the bounds, got {got}"


def test_small_noise_passes_through_unclipped():
    got = np.asarray(smoothed_action(np.array([0.1]), np.array([1.0]), 0.2, 0.5, -1.0, 1.0))
    assert np.allclose(got, [0.3]), f"expected 0.1 + 0.2 = 0.3, got {got}"


def test_zero_noise_returns_the_target_action():
    mu = np.array([0.3, -0.4])
    got = np.asarray(smoothed_action(mu, np.array([5.0, -5.0]), 0.0, 0.5, -1.0, 1.0))
    assert np.allclose(got, mu), f"sigma = 0 must leave {mu} unchanged, got {got}"


def test_array_bounds_and_batch_shape():
    mu = np.zeros((4, 2))
    got = np.asarray(smoothed_action(mu, np.full((4, 2), 3.0), 0.2, 0.5, np.array([-1.0, -1.0]), np.array([0.3, 1.0])))
    assert got.shape == (4, 2), f"expected shape (4, 2), got {got.shape}"
    assert np.allclose(got[:, 0], 0.3) and np.allclose(got[:, 1], 0.5), f"per-dimension bounds ignored: {got[0]}"


def test_sampler_matches_the_deterministic_function():
    mu = np.array([0.1, 0.2, 0.3])
    got = np.asarray(sample_smoothed_action(mu, 0.2, 0.5, -1.0, 1.0, np.random.default_rng(7)))
    z = np.random.default_rng(7).standard_normal(3)
    want = np.clip(mu + np.clip(0.2 * z, -0.5, 0.5), -1.0, 1.0)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_sampler_stays_within_c_of_the_target_action():
    mu = np.zeros(20000)
    got = np.asarray(sample_smoothed_action(mu, 1.0, 0.5, -1.0, 1.0, np.random.default_rng(0)))
    assert np.abs(got).max() <= 0.5 + 1e-12, f"max |noise| must be at most c = 0.5, got {np.abs(got).max()}"
    assert np.isclose(np.abs(got).max(), 0.5), f"with sigma = 1 many draws should hit the clip, got {np.abs(got).max()}"
