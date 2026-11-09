import numpy as np
from submission import verify_draft

V = 4


def onehot(i):
    v = np.zeros(V)
    v[i] = 1.0
    return v


def build():
    """gamma = 3. Residuals are one-hot with different tokens, so the test can see which index was used."""
    p = np.array([
        [0.5, 0.5, 0.0, 0.0],  # p_1: guess 0 (token 0) has p/q = 1.0
        [0.2, 0.3, 0.5, 0.0],  # p_2: guess 1 (token 0) has p/q = 0.25; residual picks token 2
        [0.1, 0.1, 0.1, 0.7],  # p_3: guess 2 (token 1) has p/q = 0.2; residual picks token 3
        onehot(3),             # p_4: the extra token when everything is accepted
    ])
    q = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.1, 0.5, 0.2, 0.2],
    ])
    draft = np.array([0, 0, 1])
    return p, q, draft


def test_all_accepted_adds_a_bonus_token():
    p, q, draft = build()
    out = verify_draft(p, q, draft, np.array([0.0, 0.0, 0.0]), np.random.default_rng(0))
    assert out == [0, 0, 1, 3], f"expected [0, 0, 1, 3] (3 drafts + the token from p_4), got {out}"


def test_first_guess_rejected():
    p = np.array([[0.2, 0.8, 0.0, 0.0], onehot(3)])
    q = np.array([[0.8, 0.2, 0.0, 0.0]])
    out = verify_draft(p, q, np.array([0]), np.array([0.9]), np.random.default_rng(0))
    assert out == [1], f"p/q = 0.25 < r = 0.9 rejects the guess; the residual is token 1, expected [1], got {out}"


def test_reject_at_second_guess():
    p, q, draft = build()
    out = verify_draft(p, q, draft, np.array([0.1, 0.9, 0.0]), np.random.default_rng(0))
    assert out == [0, 2], f"expected [0, 2]: guess 1 kept, guess 2 replaced by the residual token 2, got {out}"


def test_reject_at_third_guess():
    p, q, draft = build()
    out = verify_draft(p, q, draft, np.array([0.1, 0.1, 0.9]), np.random.default_rng(0))
    assert out == [0, 0, 3], f"expected [0, 0, 3]: two guesses kept, then the residual token 3, got {out}"


def test_stops_at_first_rejection():
    p, q, draft = build()
    # Guess 2 is rejected (0.9 > 0.25). Guess 3 would pass (0.0 <= 0.2), but it is not reachable.
    out = verify_draft(p, q, draft, np.array([0.1, 0.9, 0.0]), np.random.default_rng(0))
    assert len(out) == 2, f"expected 2 tokens, got {len(out)}: {out}. Guesses after a rejection must be discarded"


def test_accept_when_target_is_larger():
    p = np.array([[0.1, 0.9, 0.0, 0.0], onehot(2)])
    q = np.array([[0.7, 0.3, 0.0, 0.0]])
    out = verify_draft(p, q, np.array([1]), np.array([0.999]), np.random.default_rng(0))
    assert out == [1, 2], f"p/q = 3 must accept for any r, expected [1, 2], got {out}"


def test_returns_python_ints():
    p, q, draft = build()
    out = verify_draft(p, q, draft, np.array([0.0, 0.0, 0.0]), np.random.default_rng(0))
    assert isinstance(out, list) and all(isinstance(t, int) for t in out), f"expected a list of ints, got {out!r}"


def simulate(alpha_p, alpha_q, gamma, trials, seed):
    """The same p and q at every position, so the beta values are i.i.d. and Equation 1 applies."""
    rng = np.random.default_rng(seed)
    p_dists = np.tile(alpha_p, (gamma + 1, 1))
    q_dists = np.tile(alpha_q, (gamma, 1))
    firsts = np.zeros(len(alpha_p))
    lengths = 0
    for _ in range(trials):
        draft = rng.choice(len(alpha_q), size=gamma, p=alpha_q)
        out = verify_draft(p_dists, q_dists, draft, rng.random(gamma), rng)
        firsts[out[0]] += 1
        lengths += len(out)
    return firsts, lengths / trials


def test_first_token_follows_target():
    p = np.array([0.5, 0.4, 0.1, 0.0])
    q = np.array([0.2, 0.3, 0.5, 0.0])
    firsts, _ = simulate(p, q, gamma=3, trials=6000, seed=1)
    expected = p * 6000
    mask = expected > 0
    stat = float(np.sum((firsts[mask] - expected[mask]) ** 2 / expected[mask]))
    assert stat < 13.82, f"chi-square {stat:.2f} exceeds 13.82; counts {firsts}, expected {expected}"


def test_mean_length_matches_equation_1():
    p = np.array([0.5, 0.4, 0.1, 0.0])
    q = np.array([0.2, 0.3, 0.5, 0.0])
    alpha, gamma = 0.6, 4
    want = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    _, mean_len = simulate(p, q, gamma=gamma, trials=6000, seed=2)
    assert abs(mean_len - want) < 0.08, f"expected a mean of {want:.3f} tokens per run (Equation 1), got {mean_len:.3f}"
