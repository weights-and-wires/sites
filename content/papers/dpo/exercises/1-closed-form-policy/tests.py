import numpy as np
from submission import optimal_policy, implicit_reward

REF = np.array([0.70, 0.25, 0.05])
R = np.array([0.0, 1.0, 2.0])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_matches_page_example_beta_1():
    pi, log_Z = optimal_policy(REF, R, 1.0)
    want = np.array([0.4002, 0.3885, 0.2112])
    assert np.allclose(pi, want, atol=1e-4), f"expected pi {want}, got {pi}"
    assert abs(log_Z - np.log(1.7490)) < 1e-3, f"expected log Z {np.log(1.7490):.4f} (Z = 1.7490), got {log_Z}"


def test_matches_page_example_beta_half():
    pi, log_Z = optimal_policy(REF, R, 0.5)
    want = np.array([0.1326, 0.3500, 0.5173])
    assert np.allclose(pi, want, atol=1e-4), f"expected pi {want}, got {pi}"
    assert abs(np.exp(log_Z) - 5.2772) < 1e-3, f"expected Z 5.2772, got {np.exp(log_Z)}"


def test_is_a_distribution_and_log_Z_is_a_float():
    pi, log_Z = optimal_policy(REF, R, 0.3)
    assert abs(pi.sum() - 1.0) < 1e-9, f"pi must sum to 1, got {pi.sum()}"
    assert np.all(pi >= 0), f"pi must be non-negative, got {pi}"
    assert isinstance(log_Z, float), f"log_Z must be a float, got {type(log_Z)}"


def test_zero_reward_returns_the_reference():
    pi, log_Z = optimal_policy(REF, np.zeros(3), 0.7)
    assert np.allclose(pi, REF), f"with r = 0 the optimum is pi_ref {REF}, got {pi}"
    assert abs(log_Z) < 1e-9, f"with r = 0, Z = 1 and log Z = 0, got {log_Z}"


def test_beta_scales_r_by_division():
    big, _ = optimal_policy(REF, R, 1000.0)
    small, _ = optimal_policy(REF, R, 0.01)
    assert np.allclose(big, REF, atol=5e-3), f"a huge beta should keep pi near pi_ref {REF}, got {big}"
    assert small[2] > 0.999, f"a tiny beta should put the mass on the best response, got {small}"


def test_reference_probabilities_matter():
    # Two responses with the same reward: the optimum keeps the reference ratio.
    pi, _ = optimal_policy(np.array([0.9, 0.1]), np.array([1.0, 1.0]), 0.5)
    assert np.allclose(pi, [0.9, 0.1]), f"equal rewards must keep the reference ratio [0.9, 0.1], got {pi}"


def test_constant_shift_changes_log_Z_but_not_pi():
    pi1, z1 = optimal_policy(REF, R, 1.0)
    pi2, z2 = optimal_policy(REF, R + 5.0, 1.0)
    assert np.allclose(pi1, pi2), f"Lemma 2: pi must not change under r + f(x); got {pi1} and {pi2}"
    assert abs((z2 - z1) - 5.0) < 1e-9, f"log Z must grow by 5 / beta = 5, grew by {z2 - z1}"


def test_stable_for_large_rewards():
    pi, log_Z = optimal_policy(np.array([0.5, 0.3, 0.2]), np.array([0.0, 1000.0, 2000.0]), 1.0)
    assert np.all(np.isfinite(pi)) and np.isfinite(log_Z), f"expected finite values, got pi {pi}, log_Z {log_Z}"
    assert np.allclose(pi, [0, 0, 1]), f"expected [0, 0, 1], got {pi}"
    assert abs(log_Z - (2000.0 + np.log(0.2))) < 1e-6, f"expected log Z {2000.0 + np.log(0.2)}, got {log_Z}"


def test_implicit_reward_values():
    pi, _ = optimal_policy(REF, R, 1.0)
    got = implicit_reward(pi, REF, 1.0)
    want = np.array([-0.5591, 0.4409, 1.4409])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_equation_5_recovers_the_reward():
    for beta in (0.5, 1.0, 2.0):
        pi, log_Z = optimal_policy(REF, R, beta)
        back = implicit_reward(pi, REF, beta) + beta * log_Z
        assert np.allclose(back, R, atol=1e-9), f"beta={beta}: Equation 5 expected {R}, got {back}"


def test_partition_function_cancels_in_preferences():
    # Equation 6: p(y1 > y2) = sigma(rhat(y1) - rhat(y2)), the same as sigma(r(y1) - r(y2)).
    for beta in (0.5, 1.0):
        pi, _ = optimal_policy(REF, R, beta)
        rhat = implicit_reward(pi, REF, beta)
        got = sigmoid(rhat[2] - rhat[0])
        want = sigmoid(R[2] - R[0])
        assert abs(got - want) < 1e-9, f"beta={beta}: expected p = {want:.4f}, got {got:.4f}"
