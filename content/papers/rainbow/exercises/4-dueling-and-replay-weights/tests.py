import numpy as np
from submission import dueling_distribution, importance_weights, replay_probabilities

v = np.array([0.0, 1.0, 0.0])
adv = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, 0.0]])


def test_dueling_page_example():
    got = np.asarray(dueling_distribution(v, adv))
    want = np.array([[0.5065, 0.3072, 0.1863], [0.0629, 0.7662, 0.1710]])
    assert got.shape == (2, 3), f"expected shape (2, 3), got {got.shape}"
    assert np.allclose(got, want, atol=1e-3), f"expected {want}, got {got}"


def test_dueling_rows_sum_to_one():
    got = np.asarray(dueling_distribution(v, adv))
    assert np.allclose(got.sum(axis=1), 1.0), f"every action's distribution must sum to 1, got {got.sum(axis=1)}"


def test_dueling_ignores_a_shared_advantage_offset():
    shifted = adv + np.array([0.7, -1.2, 3.0])[None, :]
    a = np.asarray(dueling_distribution(v, adv))
    b = np.asarray(dueling_distribution(v, shifted))
    assert np.allclose(a, b), f"adding the same per-atom offset to every action must change nothing: expected {a}, got {b}"


def test_dueling_mean_over_actions_not_atoms():
    adv2 = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    got = np.asarray(dueling_distribution(np.zeros(3), adv2))
    # Per atom the mean over the two actions is [1.5, 0, 0], so action 0 has logits [1.5, 0, 0].
    want0 = np.exp([1.5, 0.0, 0.0]) / np.exp([1.5, 0.0, 0.0]).sum()
    assert np.allclose(got[0], want0), f"expected {want0}, got {got[0]}"


def test_dueling_large_logits_do_not_overflow():
    got = np.asarray(dueling_distribution(np.array([1000.0, 0.0]), np.zeros((2, 2))))
    assert np.all(np.isfinite(got)), f"a large value stream should not overflow, got {got}"
    assert np.allclose(got[:, 0], 1.0), f"all mass sits on atom 0, got {got}"


def test_replay_probabilities_page_example():
    got = np.asarray(replay_probabilities([0.04, 0.25, 1.0, 0.16], 0.5))
    want = np.array([0.0952, 0.2381, 0.4762, 0.1905])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_replay_probabilities_omega_extremes():
    got = np.asarray(replay_probabilities([1.0, 3.0], 0.0))
    assert np.allclose(got, [0.5, 0.5]), f"omega = 0 is uniform, got {got}"
    got = np.asarray(replay_probabilities([1.0, 3.0], 1.0))
    assert np.allclose(got, [0.25, 0.75]), f"omega = 1 is proportional, got {got}"


def test_importance_weights_page_example():
    probs = np.array([0.0952, 0.2381, 0.4762, 0.1905])
    got = np.asarray(importance_weights(probs, 1.0))
    assert np.allclose(got, [1.0, 0.4, 0.2, 0.5], atol=2e-3), f"expected [1, 0.4, 0.2, 0.5], got {got}"
    got = np.asarray(importance_weights(probs, 0.4))
    assert np.allclose(got, [1.0, 0.6931, 0.5253, 0.7579], atol=2e-3), f"expected [1, 0.6931, 0.5253, 0.7579], got {got}"


def test_importance_weights_uniform_and_beta_zero():
    got = np.asarray(importance_weights(np.full(4, 0.25), 1.0))
    assert np.allclose(got, 1.0), f"uniform sampling needs no correction, got {got}"
    got = np.asarray(importance_weights(np.array([0.1, 0.9]), 0.0))
    assert np.allclose(got, 1.0), f"beta = 0 means no correction, got {got}"


def test_importance_weights_max_is_one_for_the_rarest_item():
    got = np.asarray(importance_weights(np.array([0.7, 0.2, 0.1]), 0.6))
    assert np.isclose(got.max(), 1.0) and int(np.argmax(got)) == 2, f"the rarest item gets weight 1, got {got}"
