import numpy as np
from submission import cec_jacobians, lag_below, rnn_jacobians


def run_rnn(W, U, xs, h0):
    h = h0
    for x in xs:
        h = np.tanh(W @ h + U @ x)
    return h


def setup(seed=0, n=5, T=6, scale=1.0):
    rng = np.random.default_rng(seed)
    W = scale * rng.normal(size=(n, n)) / np.sqrt(n)
    U = rng.normal(size=(n, 2))
    xs = rng.normal(size=(T, 2))
    h0 = 0.3 * rng.normal(size=n)
    return W, U, xs, h0


def test_shape():
    W, U, xs, h0 = setup()
    J = rnn_jacobians(W, U, xs, h0)
    assert J.shape == (6, 5, 5), f"expected shape (6, 5, 5), got {J.shape}"


def test_first_step_matches_formula():
    W, U, xs, h0 = setup()
    h1 = np.tanh(W @ h0 + U @ xs[0])
    want = np.diag(1 - h1**2) @ W
    got = rnn_jacobians(W, U, xs, h0)[0]
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_matches_finite_differences_at_every_lag():
    W, U, xs, h0 = setup(seed=1)
    J = rnn_jacobians(W, U, xs, h0)
    eps = 1e-6
    for t in (1, 3, 6):
        fd = np.zeros_like(W)
        for k in range(W.shape[0]):
            d = np.zeros(W.shape[0])
            d[k] = eps
            hp = run_rnn(W, U, xs[:t], h0 + d)
            hm = run_rnn(W, U, xs[:t], h0 - d)
            fd[:, k] = (hp - hm) / (2 * eps)
        assert np.allclose(J[t - 1], fd, atol=1e-6), f"lag {t}: expected\n{fd}\ngot\n{J[t - 1]}"


def test_small_weights_decay_like_equation_3_2():
    # With ||W||_2 <= 0.5 and tanh' <= 1, every step shrinks the error by at least 2.
    rng = np.random.default_rng(2)
    n, T = 8, 30
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    W = 0.5 * Q
    U = rng.normal(size=(n, 2))
    xs = rng.normal(size=(T, 2))
    J = rnn_jacobians(W, U, xs, np.zeros(n))
    norms = np.array([np.linalg.norm(Jt, 2) for Jt in J])
    bound = 0.5 ** np.arange(1, T + 1)
    assert np.all(norms <= bound + 1e-12), f"norms must be at most 0.5^q; got {norms[:4]} against {bound[:4]}"
    assert norms[-1] < 1e-8, f"after {T} steps the norm should be tiny, got {norms[-1]}"
    assert np.all(norms < 0.5 ** np.arange(1, T + 1) * 0.999), f"saturation must shrink the norm below the bound, got {norms[:4]}"


def test_saturation_shrinks_error_further():
    # Large inputs saturate tanh, so the factor 1 - h^2 is near 0 and the error dies faster.
    n = 4
    W = np.eye(n)
    U = np.ones((n, 1))
    small = rnn_jacobians(W, U, 0.01 * np.ones((5, 1)), np.zeros(n))[-1]
    large = rnn_jacobians(W, U, 5.0 * np.ones((5, 1)), np.zeros(n))[-1]
    assert np.linalg.norm(small, 2) > 0.9, f"unsaturated unit with w = 1 should keep error, got {np.linalg.norm(small, 2)}"
    assert np.linalg.norm(large, 2) < 1e-3, f"saturated units should lose error, got {np.linalg.norm(large, 2)}"


def test_cec_keeps_error_exactly():
    J = cec_jacobians(3, 1000)
    assert J.shape == (1000, 3, 3), f"expected shape (1000, 3, 3), got {J.shape}"
    assert np.array_equal(J[-1], np.eye(3)), f"expected the identity after 1000 steps, got\n{J[-1]}"
    assert np.allclose(J[0], np.eye(3)), f"expected the identity after 1 step, got\n{J[0]}"


def test_lag_below():
    norms = np.array([1.0, 0.5, 0.09, 0.01])
    assert lag_below(norms, 0.1) == 3, f"expected 3, got {lag_below(norms, 0.1)}"
    assert lag_below(norms, 0.5) == 3, f"threshold is strict; expected 3, got {lag_below(norms, 0.5)}"
    assert lag_below(norms, 1e-6) is None, f"expected None, got {lag_below(norms, 1e-6)}"
    assert lag_below(np.array([0.2, 0.9]), 0.5) == 1, "lag counts from 1"


def test_rnn_vanishes_and_cec_does_not():
    W, U, xs, h0 = setup(seed=3, n=16, T=100)
    norms = [np.linalg.norm(J, 2) for J in rnn_jacobians(W, U, xs, h0)]
    cec = [np.linalg.norm(J, 2) for J in cec_jacobians(16, 100)]
    assert lag_below(norms, 1e-6) is not None, f"the RNN norm at lag 100 is {norms[-1]}; it should fall below 1e-6"
    assert lag_below(cec, 0.999) is None, f"the CEC norm must stay at 1, got min {min(cec)}"
