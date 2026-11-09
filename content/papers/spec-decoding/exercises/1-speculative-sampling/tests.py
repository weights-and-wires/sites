import numpy as np
from submission import chi_square_stat, residual_distribution, speculative_sample

P = np.array([0.5, 0.4, 0.1])
Q = np.array([0.2, 0.3, 0.5])
N = 10_000
# Chi-square critical values at the 0.001 level, by degrees of freedom.
CRIT = {1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47}


def run(p, q, seed, n=N):
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(p))
    acc = 0
    for _ in range(n):
        tok, ok = speculative_sample(p, q, rng)
        counts[tok] += 1
        acc += bool(ok)
    return counts, acc / n


def test_residual_values():
    got = residual_distribution(P, Q)
    want = np.array([0.75, 0.25, 0.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_residual_is_a_distribution():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([0.4, 0.3, 0.2, 0.1])
    got = residual_distribution(p, q)
    assert np.all(got >= 0), f"entries must be non-negative, got {got}"
    assert np.isclose(got.sum(), 1.0), f"expected a sum of 1, got {got.sum()}"
    assert np.allclose(got, [0, 0, 0.25, 0.75]), f"expected [0, 0, 0.25, 0.75], got {got}"


def test_output_matches_target():
    counts, _ = run(P, Q, seed=1)
    stat = chi_square_stat(counts, P)
    assert stat < CRIT[2], f"chi-square {stat:.2f} exceeds {CRIT[2]}; counts {counts}, expected {P * N}"


def test_output_matches_target_other_pair():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([0.4, 0.3, 0.2, 0.1])
    counts, _ = run(p, q, seed=2)
    stat = chi_square_stat(counts, p)
    assert stat < CRIT[3], f"chi-square {stat:.2f} exceeds {CRIT[3]}; counts {counts}, expected {p * N}"


def test_acceptance_rate_is_beta():
    _, rate = run(P, Q, seed=3)
    beta = np.minimum(P, Q).sum()
    assert abs(rate - beta) < 0.03, f"expected an acceptance rate near {beta:.3f}, got {rate:.3f}"


def test_identical_models_always_accept():
    counts, rate = run(P, P, seed=4, n=2000)
    assert rate == 1.0, f"when q == p every draft token must be accepted, got rate {rate}"
    assert chi_square_stat(counts, P) < CRIT[2], f"output must follow p, got counts {counts}"


def test_draft_that_never_proposes_a_token():
    # q puts no mass on token 2, so every sample of token 2 must come from the residual.
    p = np.array([0.3, 0.3, 0.4])
    q = np.array([0.5, 0.5, 0.0])
    counts, _ = run(p, q, seed=5)
    stat = chi_square_stat(counts, p)
    assert counts[2] > 0, f"token 2 must be reachable through the residual, counts {counts}"
    assert stat < CRIT[2], f"chi-square {stat:.2f} exceeds {CRIT[2]}; counts {counts}, expected {p * N}"
