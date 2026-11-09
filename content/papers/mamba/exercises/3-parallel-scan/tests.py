import numpy as np
from submission import combine, parallel_scan


def sequential(a, b):
    h = np.zeros(a.shape[1:])
    out = []
    for t in range(a.shape[0]):
        h = a[t] * h + b[t]
        out.append(h)
    return np.stack(out)


def test_combine_order():
    p, q = (0.5, 1.0), (0.25, 2.0)
    got = combine(p, q)
    assert np.allclose(got, (0.125, 2.25)), f"combine(p, q): expected (0.125, 2.25), got {got}"
    got = combine(q, p)
    assert np.allclose(got, (0.125, 2.0)), f"combine(q, p): expected (0.125, 2.0), got {got}"


def test_combine_is_associative():
    rng = np.random.default_rng(0)
    x, y, z = [tuple(rng.normal(size=2)) for _ in range(3)]
    lhs, rhs = combine(combine(x, y), z), combine(x, combine(y, z))
    assert np.allclose(lhs, rhs), f"(x.y).z = {lhs} but x.(y.z) = {rhs}"


def test_hand_example():
    a = np.full(4, 0.5)
    b = np.array([0.5, 1.0, 1.5, 2.0])
    h = parallel_scan(a, b)
    assert np.allclose(h, [0.5, 1.25, 2.125, 3.0625]), f"expected [0.5, 1.25, 2.125, 3.0625], got {h}"


def test_matches_sequential_for_awkward_lengths():
    rng = np.random.default_rng(1)
    for L in [1, 2, 3, 5, 7, 8, 9, 31, 100]:
        a = rng.uniform(0.3, 1.1, size=L)
        b = rng.normal(size=L)
        want = sequential(a, b)
        got = parallel_scan(a, b)
        assert got.shape == want.shape, f"L={L}: shape expected {want.shape}, got {got.shape}"
        assert np.allclose(got, want), f"L={L}: expected {want}, got {got}"


def test_state_axes_are_carried():
    rng = np.random.default_rng(2)
    a = rng.uniform(0.3, 1.0, size=(11, 3, 4))
    b = rng.normal(size=(11, 3, 4))
    want = sequential(a, b)
    got = parallel_scan(a, b)
    assert got.shape == (11, 3, 4), f"expected shape (11, 3, 4), got {got.shape}"
    assert np.allclose(got, want), "per-channel, per-state-entry recurrences must scan independently along axis 0"


def test_selective_reset():
    # a_t = 0 wipes the history, like a large delta: h_t = b_t from that step on.
    a = np.array([0.9, 0.9, 0.0, 0.9, 0.9])
    b = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    got = parallel_scan(a, b)
    want = [1.0, 1.9, 2.0, 2.8, 3.52]
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_inputs_not_modified():
    a = np.array([0.5, 0.5, 0.5, 0.5])
    b = np.array([0.5, 1.0, 1.5, 2.0])
    a0, b0 = a.copy(), b.copy()
    parallel_scan(a, b)
    assert np.array_equal(a, a0) and np.array_equal(b, b0), f"parallel_scan changed its inputs: a={a}, b={b}"
