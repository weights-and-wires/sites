import numpy as np
from submission import acceptance_rate, expected_tokens, walltime_speedup, best_gamma


def test_acceptance_rate_example():
    got = acceptance_rate([0.5, 0.4, 0.1], [0.2, 0.3, 0.5])
    assert abs(got - 0.6) < 1e-9, f"expected 0.6, got {got}"


def test_acceptance_rate_bounds():
    assert abs(acceptance_rate([1, 0, 0], [0, 1, 0])) < 1e-12, "disjoint supports must give alpha = 0"
    assert abs(acceptance_rate([0.2, 0.8], [0.2, 0.8]) - 1) < 1e-12, "identical distributions must give alpha = 1"


def test_acceptance_rate_is_symmetric():
    p, q = [0.7, 0.2, 0.1], [0.1, 0.3, 0.6]
    assert abs(acceptance_rate(p, q) - acceptance_rate(q, p)) < 1e-12, "alpha must be symmetric in p and q"


def test_expected_tokens_values():
    assert abs(expected_tokens(0.8, 5) - 3.6893) < 1e-3, f"expected 3.6893, got {expected_tokens(0.8, 5)}"
    assert abs(expected_tokens(0.5, 1) - 1.5) < 1e-12, f"expected 1.5, got {expected_tokens(0.5, 1)}"
    assert abs(expected_tokens(0.0, 7) - 1.0) < 1e-12, f"alpha = 0 always yields exactly 1 token, got {expected_tokens(0.0, 7)}"


def test_expected_tokens_alpha_one():
    got = expected_tokens(1.0, 6)
    assert got == 7, f"alpha = 1 accepts all 6 guesses plus the extra token: expected 7, got {got}"


def test_expected_tokens_is_the_geometric_sum():
    a, g = 0.65, 4
    want = sum(a**k for k in range(g + 1))
    assert abs(expected_tokens(a, g) - want) < 1e-9, f"expected {want}, got {expected_tokens(a, g)}"


def test_table_1_speed_column():
    # Table 1 of the paper, with c = 0.
    for alpha, gamma, want in [(0.6, 2, 1.96), (0.7, 3, 2.53), (0.8, 2, 2.44), (0.8, 5, 3.69), (0.9, 2, 2.71), (0.9, 10, 6.86)]:
        got = walltime_speedup(alpha, gamma, 0.0)
        assert abs(got - want) < 0.01, f"alpha {alpha}, gamma {gamma}: expected {want}, got {got:.3f}"


def test_cost_coefficient_matters():
    got = walltime_speedup(0.8, 5, 0.05)
    assert abs(got - 2.9514) < 1e-3, f"expected 2.9514 for alpha 0.8, gamma 5, c 0.05, got {got}"


def test_table_4_expected_column():
    for alpha, c, gamma, want in [(0.75, 0.02, 7, 3.2), (0.8, 0.04, 7, 3.3), (0.82, 0.11, 7, 2.5), (0.65, 0.02, 5, 2.4), (0.74, 0.11, 3, 2.0)]:
        got = walltime_speedup(alpha, gamma, c)
        assert round(got, 1) == want, f"alpha {alpha}, c {c}, gamma {gamma}: expected {want}, got {got:.3f}"


def test_gamma_one_matches_corollary_3_9():
    a, c = 0.3, 0.2
    got = walltime_speedup(a, 1, c)
    assert abs(got - (1 + a) / (1 + c)) < 1e-12, f"expected {(1 + a) / (1 + c)}, got {got}"


def test_best_gamma_grid():
    # The same grid as scripts/speedup_example.py.
    cases = [(0.5, 0.01, 5), (0.7, 0.05, 6), (0.8, 0.05, 8), (0.9, 0.01, 24), (0.9, 0.1, 10)]
    for alpha, c, want in cases:
        got = best_gamma(alpha, c)
        assert got == want, f"alpha {alpha}, c {c}: expected gamma {want}, got {got}"


def test_best_gamma_when_drafting_is_free():
    got = best_gamma(0.5, 0.0, max_gamma=12)
    assert got == 12, f"with c = 0 more guesses never hurt, so expected the cap 12, got {got}"


def test_slow_draft_prefers_one_guess():
    got = best_gamma(0.2, 0.3)
    assert got == 1, f"alpha 0.2, c 0.3: expected gamma 1, got {got}"
    assert walltime_speedup(0.2, 1, 0.3) < 1, "this setting is slower than the baseline"


def test_best_gamma_tie_goes_to_the_smaller_value():
    got = best_gamma(0.0, 0.0, max_gamma=10)
    assert got == 1, f"alpha 0 and c 0 give speedup 1 for every gamma; expected the smaller gamma 1, got {got}"
