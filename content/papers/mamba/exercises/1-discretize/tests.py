import numpy as np
from submission import discretize


def one(delta, A, B):
    a, b = discretize(np.array([[delta]]), np.array([[A]]), np.array([[B]]))
    return float(a[0, 0, 0]), float(b[0, 0, 0])


def test_shapes():
    rng = np.random.default_rng(0)
    L, D, N = 5, 3, 4
    a, b = discretize(rng.uniform(0.1, 1, (L, D)), -rng.uniform(0.5, 2, (D, N)), rng.normal(size=(L, N)))
    assert a.shape == (L, D, N), f"A_bar shape: expected {(L, D, N)}, got {a.shape}"
    assert b.shape == (L, D, N), f"B_bar shape: expected {(L, D, N)}, got {b.shape}"


def test_hand_example():
    # A = -1, B = 1, delta = ln 2: A_bar = 1/2 and B_bar = (1/2 - 1) / -1 = 1/2.
    a, b = one(np.log(2.0), -1.0, 1.0)
    assert np.isclose(a, 0.5), f"A_bar: expected 0.5, got {a}"
    assert np.isclose(b, 0.5), f"B_bar: expected 0.5, got {b}"


def test_a_is_not_minus_one():
    # With A = -2, delta = 0.5, B = 3: A_bar = exp(-1), B_bar = (exp(-1) - 1) / -2 * 3.
    a, b = one(0.5, -2.0, 3.0)
    want_a, want_b = np.exp(-1.0), (np.exp(-1.0) - 1.0) / -2.0 * 3.0
    assert np.isclose(a, want_a), f"A_bar: expected {want_a:.6f}, got {a:.6f}; it is exp(delta * A)"
    assert np.isclose(b, want_b), f"B_bar: expected {want_b:.6f}, got {b:.6f}; it is (A_bar - 1) / A * B"


def test_theorem_1_gate():
    # Theorem 1: N = 1, A = -1, B = 1, delta = softplus(z) gives A_bar = 1 - g, B_bar = g, g = sigmoid(z).
    for z in [-4.0, 0.0, 4.0]:
        g = 1 / (1 + np.exp(-z))
        a, b = one(np.log1p(np.exp(z)), -1.0, 1.0)
        assert np.isclose(a, 1 - g), f"z={z}: A_bar expected {1 - g:.6f}, got {a:.6f}"
        assert np.isclose(b, g), f"z={z}: B_bar expected {g:.6f}, got {b:.6f}"


def test_small_step_is_euler():
    # As delta -> 0, B_bar -> delta * B.
    a, b = one(1e-6, -1.0, 2.0)
    assert np.isclose(a, 1.0, atol=1e-5), f"A_bar should be close to 1 for a tiny step, got {a}"
    assert np.isclose(b, 2e-6, rtol=1e-3), f"B_bar should be close to delta * B = 2e-06, got {b}"


def test_large_step_resets_state():
    a, b = one(30.0, -1.0, 1.0)
    assert a < 1e-12, f"A_bar should be close to 0 for a large step, got {a}"
    assert np.isclose(b, 1.0), f"B_bar should approach -B/A = 1, got {b}"


def test_axes_are_not_mixed():
    rng = np.random.default_rng(3)
    L, D, N = 4, 3, 2
    delta = rng.uniform(0.1, 1.0, (L, D))
    A = -rng.uniform(0.5, 2.0, (D, N))
    B = rng.normal(size=(L, N))
    a, b = discretize(delta, A, B)
    for t in range(L):
        for d in range(D):
            for n in range(N):
                wa = np.exp(delta[t, d] * A[d, n])
                wb = (wa - 1) / A[d, n] * B[t, n]
                assert np.isclose(a[t, d, n], wa), f"A_bar[{t},{d},{n}]: expected {wa:.6f}, got {a[t, d, n]:.6f}"
                assert np.isclose(b[t, d, n], wb), f"B_bar[{t},{d},{n}]: expected {wb:.6f}, got {b[t, d, n]:.6f}"
