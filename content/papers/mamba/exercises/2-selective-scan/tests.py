import numpy as np
from submission import selective_step, selective_scan


def problem(seed=0, L=6, D=3, N=4):
    rng = np.random.default_rng(seed)
    return dict(
        x=rng.normal(size=(L, D)),
        delta=rng.uniform(0.05, 1.0, (L, D)),
        A=-rng.uniform(0.5, 2.0, (D, N)),
        B=rng.normal(size=(L, N)),
        C=rng.normal(size=(L, N)),
    )


def reference(x, delta, A, B, C):
    """Per-channel, per-state-entry loops that follow Equations 2 and 4 directly."""
    L, D = x.shape
    N = A.shape[1]
    y = np.zeros((L, D))
    for d in range(D):
        h = np.zeros(N)
        for t in range(L):
            for n in range(N):
                a_bar = np.exp(delta[t, d] * A[d, n])
                b_bar = (a_bar - 1) / A[d, n] * B[t, n]
                h[n] = a_bar * h[n] + b_bar * x[t, d]
            y[t, d] = sum(C[t, n] * h[n] for n in range(N))
    return y


def test_shapes():
    p = problem()
    y = selective_scan(**p)
    assert y.shape == (6, 3), f"y shape: expected (6, 3), got {y.shape}"
    y1, h1 = selective_step(np.zeros((3, 4)), p["x"][0], p["delta"][0], p["A"], p["B"][0], p["C"][0])
    assert y1.shape == (3,), f"y_t shape: expected (3,), got {y1.shape}"
    assert h1.shape == (3, 4), f"h shape: expected (3, 4), got {h1.shape}"


def test_matches_reference():
    p = problem(seed=1)
    want = reference(**p)
    got = selective_scan(**p)
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_hand_example():
    # N = 1, A = -1, B = C = 1, delta = ln 2 for every step, x = 1, 2, 3, 4: y = 0.5, 1.25, 2.125, 3.0625.
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    delta = np.full((4, 1), np.log(2.0))
    A = np.array([[-1.0]])
    B = np.ones((4, 1))
    C = np.ones((4, 1))
    y = selective_scan(x, delta, A, B, C)
    assert np.allclose(y[:, 0], [0.5, 1.25, 2.125, 3.0625]), f"expected [0.5, 1.25, 2.125, 3.0625], got {y[:, 0]}"


def test_output_uses_the_new_state():
    # One step from h = 0 with x = 1: h_1 = B_bar, so y_1 = C * B_bar != 0.
    A = np.array([[-1.0]])
    y, h = selective_step(np.zeros((1, 1)), np.array([1.0]), np.array([np.log(2.0)]), A, np.array([1.0]), np.array([1.0]))
    assert np.allclose(h, [[0.5]]), f"expected h_1 = [[0.5]], got {h}"
    assert np.allclose(y, [0.5]), f"expected y_1 = [0.5] (read the state after the update), got {y}"


def test_c_changes_with_time():
    p = problem(seed=2)
    y_a = selective_scan(**p)
    p2 = dict(p, C=p["C"].copy())
    p2["C"][3] *= -1.0
    y_b = selective_scan(**p2)
    assert np.allclose(y_a[:3], y_b[:3]), "flipping C_3 must not change y_0..y_2"
    assert np.allclose(y_b[3], -y_a[3]), f"flipping C_3 must flip y_3, expected {-y_a[3]}, got {y_b[3]}"


def test_delta_is_per_step():
    # A tiny delta at step 2 keeps the state and ignores x_2; a large delta replaces the state with x_2.
    A = np.array([[-1.0]])
    x = np.array([[5.0], [1.0], [9.0]])
    B = np.ones((3, 1))
    C = np.ones((3, 1))
    keep = selective_scan(x, np.array([[30.0], [30.0], [1e-6]]), A, B, C)
    assert np.isclose(keep[2, 0], 1.0, atol=1e-4), f"a tiny delta must keep h = 1; expected 1.0, got {keep[2, 0]}"
    reset = selective_scan(x, np.array([[30.0], [30.0], [30.0]]), A, B, C)
    assert np.isclose(reset[2, 0], 9.0, atol=1e-4), f"a large delta must reset h to x_2 = 9; expected 9.0, got {reset[2, 0]}"


def test_lti_case_is_a_convolution():
    # Constant delta, B, and C make the model time-invariant, so Equation 3 applies.
    rng = np.random.default_rng(5)
    L, N = 8, 3
    x = rng.normal(size=(L, 1))
    A = -rng.uniform(0.5, 2.0, (1, N))
    delta = np.full((L, 1), 0.4)
    B = np.tile(rng.normal(size=(1, N)), (L, 1))
    C = np.tile(rng.normal(size=(1, N)), (L, 1))
    a_bar = np.exp(0.4 * A[0])
    b_bar = (a_bar - 1) / A[0] * B[0]
    K = np.array([(C[0] * a_bar**k * b_bar).sum() for k in range(L)])
    want = np.array([sum(K[k] * x[t - k, 0] for k in range(t + 1)) for t in range(L)])
    got = selective_scan(x, delta, A, B, C)[:, 0]
    assert np.allclose(got, want), f"Equation 3 kernel gives\n{want}\nbut the recurrence gives\n{got}"


def test_stepwise_equals_scan_with_fixed_size_state():
    p = problem(seed=4)
    full = selective_scan(**p)
    h = np.zeros(p["A"].shape)
    for t in range(p["x"].shape[0]):
        y_t, h = selective_step(h, p["x"][t], p["delta"][t], p["A"], p["B"][t], p["C"][t])
        assert h.shape == p["A"].shape, f"step {t}: the state must stay {p['A'].shape}, got {h.shape}"
        assert np.allclose(y_t, full[t]), f"step {t}: expected {full[t]}, got {y_t}"
