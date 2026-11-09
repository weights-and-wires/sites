import numpy as np
from submission import flatten_acoustic, unflatten_acoustic

N, Q = 1024, 4
Y = np.array([[5, 900, 17, 3], [6, 12, 1000, 4], [5, 7, 2, 1023]])
WANT = [5, 1924, 2065, 3075, 6, 1036, 3048, 3076, 5, 1031, 2050, 4095]


def test_example_from_page():
    got = np.asarray(flatten_acoustic(Y, N))
    assert got.tolist() == WANT, f"expected {WANT}, got {got.tolist()}"


def test_shape_and_order():
    got = np.asarray(flatten_acoustic(Y, N))
    assert got.shape == (12,), f"expected shape (12,), got {got.shape}"
    layer0 = got[0::Q] % N
    assert layer0.tolist() == [5, 6, 5], f"every 4th id should be layer 0 of steps 0..2, expected [5, 6, 5], got {layer0.tolist()}"


def test_each_layer_has_its_own_range():
    rng = np.random.default_rng(0)
    big = rng.integers(0, N, size=(50, Q))
    got = np.asarray(flatten_acoustic(big, N)).reshape(-1, Q)
    for q in range(Q):
        assert got[:, q].min() >= q * N and got[:, q].max() < (q + 1) * N, f"layer {q} ids must lie in [{q * N}, {(q + 1) * N}), got [{got[:, q].min()}, {got[:, q].max()}]"


def test_round_trip():
    rng = np.random.default_rng(1)
    for q in (1, 3, 12):
        y = rng.integers(0, N, size=(7, q))
        back = unflatten_acoustic(flatten_acoustic(y, N), q, N)
        assert back.shape == y.shape and np.array_equal(back, y), f"Q={q}: round trip changed the matrix; expected\n{y}\ngot\n{back}"


def test_unflatten_example():
    got = unflatten_acoustic(np.array(WANT), Q, N)
    assert np.array_equal(got, Y), f"expected\n{Y}\ngot\n{got}"


def test_out_of_range_symbol():
    bad = np.array([[0, 1024]])
    try:
        flatten_acoustic(bad, N)
    except ValueError:
        return
    raise AssertionError("a symbol equal to N must raise ValueError, but nothing was raised")


def test_bad_length():
    try:
        unflatten_acoustic(np.arange(10), 4, N)
    except ValueError:
        return
    raise AssertionError("a length of 10 with Q=4 must raise ValueError, but nothing was raised")
