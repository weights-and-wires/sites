import numpy as np
from submission import boltzmann_policy, soft_value

q = np.array([1.0, 2.0, 3.0])


def test_policy_on_page_example():
    pi = np.asarray(boltzmann_policy(q, 1.0))
    want = np.array([0.0900, 0.2447, 0.6652])
    assert pi.shape == (3,), f"expected shape (3,), got {pi.shape}"
    assert np.allclose(pi, want, atol=1e-4), f"expected {want}, got {pi}"
    assert np.isclose(pi.sum(), 1.0), f"probabilities must sum to 1, got {pi.sum()}"


def test_value_on_page_example():
    got = soft_value(q, 1.0)
    assert np.isclose(got, 3.4076, atol=1e-4), f"expected 3.4076, got {got}"


def test_value_matches_equation_3():
    for alpha in (0.5, 1.0, 2.0):
        pi = np.asarray(boltzmann_policy(q, alpha))
        want = np.sum(pi * (q - alpha * np.log(pi)))
        got = soft_value(q, alpha)
        assert np.isclose(got, want), f"alpha={alpha}: E[Q - alpha log pi] = {want}, got {got}"


def test_temperature_scales_the_logits():
    pi = np.asarray(boltzmann_policy(q, 0.5))
    want = np.exp(q / 0.5) / np.exp(q / 0.5).sum()
    assert np.allclose(pi, want), f"alpha=0.5: expected {want}, got {pi}"
    v = soft_value(q, 0.5)
    assert np.isclose(v, 3.0715, atol=1e-4), f"alpha=0.5: expected 3.0715, got {v}"


def test_small_alpha_approaches_the_maximum():
    got = soft_value(q, 0.01)
    assert np.isclose(got, 3.0, atol=1e-3), f"alpha=0.01 should give about max Q = 3, got {got}"
    pi = np.asarray(boltzmann_policy(q, 0.01))
    assert pi[2] > 0.999, f"alpha=0.01 should put almost all mass on the best action, got {pi}"


def test_large_alpha_approaches_uniform_plus_entropy():
    pi = np.asarray(boltzmann_policy(q, 1000.0))
    assert np.allclose(pi, 1 / 3, atol=1e-3), f"alpha=1000 should be near uniform, got {pi}"
    got = soft_value(np.array([2.0, 2.0, 2.0]), 3.0)
    want = 2.0 + 3.0 * np.log(3)
    assert np.isclose(got, want), f"equal Q: expected Q + alpha log 3 = {want}, got {got}"


def test_shift_equivariance():
    got = soft_value(q + 10.0, 1.0)
    assert np.isclose(got, 13.4076, atol=1e-4), f"adding 10 to every Q adds 10 to V: expected 13.4076, got {got}"
    pi = np.asarray(boltzmann_policy(q + 10.0, 1.0))
    assert np.allclose(pi, boltzmann_policy(q, 1.0)), f"the policy must not change when you shift Q, got {pi}"


def test_large_values_do_not_overflow():
    big = np.array([1000.0, 1001.0])
    got = soft_value(big, 1.0)
    want = 1001.0 + np.log1p(np.exp(-1.0))
    assert np.isfinite(got) and np.isclose(got, want), f"expected {want}, got {got}"
    pi = np.asarray(boltzmann_policy(big, 1.0))
    assert np.all(np.isfinite(pi)) and np.isclose(pi.sum(), 1.0), f"policy must be finite and sum to 1, got {pi}"
    assert np.isclose(pi[1], 1 / (1 + np.exp(-1.0))), f"expected {1 / (1 + np.exp(-1.0))}, got {pi[1]}"
