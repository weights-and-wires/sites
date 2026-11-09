import math
import random

from submission import online_normalizer, online_softmax


def stable(x):
    m = max(x)
    d = sum(math.exp(v - m) for v in x)
    return m, d


def test_matches_two_pass_values():
    x = [2.0, 1.0, 4.0, 3.0]
    m, d = online_normalizer(iter(x))
    assert m == 4.0, f"expected m = 4.0, got {m}"
    assert abs(d - 1.5530) < 1e-4, f"expected d = 1.5530, got {d}"


def test_reads_the_input_once():
    # A one-shot iterator: a second pass sees nothing and gets d = 0 or m = -inf.
    random.seed(0)
    x = [random.gauss(0, 3) for _ in range(40)]
    m, d = online_normalizer(iter(x))
    want_m, want_d = stable(x)
    assert abs(m - want_m) < 1e-12, f"expected m = {want_m}, got {m}"
    assert abs(d - want_d) < 1e-9, f"expected d = {want_d}, got {d}"


def test_ascending_input_rescales_every_step():
    x = [float(i) for i in range(10)]
    m, d = online_normalizer(iter(x))
    want_m, want_d = stable(x)
    assert m == want_m, f"expected m = {want_m}, got {m}"
    assert abs(d - want_d) < 1e-9, f"expected d = {want_d}, got {d}; the old sum must shrink when the max grows"


def test_descending_input_never_rescales():
    x = [float(10 - i) for i in range(10)]
    m, d = online_normalizer(iter(x))
    want_m, want_d = stable(x)
    assert m == want_m and abs(d - want_d) < 1e-9, f"expected {(want_m, want_d)}, got {(m, d)}"


def test_all_negative_inputs():
    # The running max must start at -inf, not 0.
    x = [-5.0, -3.0, -4.0]
    m, d = online_normalizer(iter(x))
    want_m, want_d = stable(x)
    assert m == want_m, f"expected m = {want_m}, got {m}"
    assert abs(d - want_d) < 1e-12, f"expected d = {want_d}, got {d}"


def test_large_values_do_not_overflow():
    x = [1000.0, 1001.0, 1002.0]
    m, d = online_normalizer(iter(x))
    assert m == 1002.0, f"expected m = 1002.0, got {m}"
    assert math.isfinite(d) and abs(d - 1.5032) < 1e-3, f"expected d = 1.5032, got {d}"


def test_single_element():
    m, d = online_normalizer(iter([7.5]))
    assert (m, d) == (7.5, 1.0), f"expected (7.5, 1.0), got {(m, d)}"


def test_normalizer_bounds():
    # Section 3: 1 <= d_V <= V.
    random.seed(1)
    for n in (1, 5, 30):
        x = [random.gauss(0, 4) for _ in range(n)]
        _, d = online_normalizer(iter(x))
        assert 1 - 1e-12 <= d <= n + 1e-12, f"expected 1 <= d <= {n}, got {d}"


def test_softmax_matches_stable_softmax():
    random.seed(2)
    x = [random.gauss(0, 20) for _ in range(25)]
    m, d = stable(x)
    want = [math.exp(v - m) / d for v in x]
    got = online_softmax(x)
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, want)), f"expected {want[:3]}..., got {got[:3]}..."
    assert abs(sum(got) - 1) < 1e-12, f"outputs must sum to 1, got {sum(got)}"
