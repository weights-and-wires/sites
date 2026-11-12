import numpy as np
from submission import verify_node

P = np.array([0.5, 0.4, 0.1])
Q = np.array([0.2, 0.3, 0.5])


def test_first_candidate_accepted():
    tok, i = verify_node(P, Q, [0, 2], [0.99, 0.0], np.random.default_rng(0))
    assert (tok, i) == (0, 0), f"p/q = 2.5 accepts token 0 for any u, expected (0, 0), got {(tok, i)}"


def test_second_candidate_is_tested_against_the_updated_target():
    # Candidate 0 is token 2 with p/q = 0.2 and u = 0.9: rejected. p becomes [0.75, 0.25, 0].
    # Candidate 1 is token 1 with ratio 0.25 / 0.3 = 0.833. With the original p the ratio would be 1.333.
    tok, i = verify_node(P, Q, [2, 1], [0.9, 0.9], np.random.default_rng(0))
    assert i == -1, f"u = 0.9 > 0.833 must reject the second candidate (updated p), got index {i}, token {tok}"


def test_second_candidate_accepted_when_u_is_small():
    tok, i = verify_node(P, Q, [2, 1], [0.9, 0.8], np.random.default_rng(0))
    assert (tok, i) == (1, 1), f"u = 0.8 <= 0.833 accepts the second candidate, expected (1, 1), got {(tok, i)}"


def test_all_rejected_samples_from_the_adjusted_target():
    # Two rejections of token 2: p -> [0.75, 0.25, 0] -> residual with q again.
    # norm(max(0, [0.75, 0.25, 0] - [0.2, 0.3, 0.5])) = [1, 0, 0].
    for seed in range(5):
        tok, i = verify_node(P, Q, [2, 2], [0.99, 0.99], np.random.default_rng(seed))
        assert (tok, i) == (0, -1), f"expected the only token with mass left, (0, -1), got {(tok, i)}"


def test_returns_python_ints():
    tok, i = verify_node(P, Q, [0], [0.0], np.random.default_rng(0))
    assert type(tok) is int and type(i) is int, f"expected ints, got {type(tok)} and {type(i)}"


def run(k, trials, seed):
    rng = np.random.default_rng(seed)
    counts = np.zeros(3)
    accepted = 0
    for _ in range(trials):
        cands = rng.choice(3, size=k, p=Q)
        tok, i = verify_node(P, Q, cands, rng.random(k), rng)
        counts[tok] += 1
        accepted += i >= 0
    return counts, accepted / trials


def chi2(counts, trials):
    want = P * trials
    return float(np.sum((counts - want) ** 2 / want))


def test_output_follows_the_target_for_k_2():
    counts, _ = run(2, 20000, 1)
    stat = chi2(counts, 20000)
    assert stat < 13.82, f"chi-square {stat:.2f} exceeds 13.82; frequencies {counts / 20000}, expected {P}"


def test_output_follows_the_target_for_k_3():
    counts, _ = run(3, 20000, 2)
    stat = chi2(counts, 20000)
    assert stat < 13.82, f"chi-square {stat:.2f} exceeds 13.82; frequencies {counts / 20000}, expected {P}"


def test_acceptance_rate_for_k_2():
    _, rate = run(2, 20000, 3)
    assert abs(rate - 0.78) < 0.02, f"expected an acceptance rate near 0.78, got {rate:.4f}"
