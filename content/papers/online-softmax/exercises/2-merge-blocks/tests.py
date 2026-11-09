import math

import numpy as np
from submission import blocked_softmax, combine


def stable_softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def close(a, b, tol=1e-10):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol * max(1.0, abs(b[1]))


def test_page_example():
    left, right = (2.0, 1.3679), (4.0, 1.3679)
    m, d = combine(left, right)
    assert m == 4.0, f"expected m = 4.0, got {m}"
    assert abs(d - 1.5530) < 1e-4, f"expected d = 1.5530, got {d}"


def test_rescales_both_sides():
    # The larger max can sit on either side.
    a, b = (0.0, 3.0), (2.0, 1.0)
    want = (2.0, 3.0 * math.exp(-2.0) + 1.0)
    assert close(combine(a, b), want), f"expected {want}, got {combine(a, b)}"
    assert close(combine(b, a), want), f"swapped: expected {want}, got {combine(b, a)}"


def test_equal_maxima_add_the_sums():
    m, d = combine((1.5, 2.0), (1.5, 3.0))
    assert m == 1.5 and abs(d - 5.0) < 1e-12, f"expected (1.5, 5.0), got {(m, d)}"


def test_commutative_and_associative():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b, c = [(float(rng.normal(0, 4)), float(rng.uniform(1, 5))) for _ in range(3)]
        assert close(combine(a, b), combine(b, a)), f"not commutative: {combine(a, b)} vs {combine(b, a)}"
        left, right = combine(combine(a, b), c), combine(a, combine(b, c))
        assert close(left, right), f"not associative: {left} vs {right}"


def test_empty_summary_is_neutral():
    s = (3.0, 1.7)
    ident = (-math.inf, 0.0)
    assert close(combine(ident, s), s), f"expected {s}, got {combine(ident, s)}"
    assert close(combine(s, ident), s), f"expected {s}, got {combine(s, ident)}"


def test_blocked_matches_stable_softmax():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 5, size=37)
    want = stable_softmax(x)
    for bs in (1, 4, 10, 37, 100):
        got = blocked_softmax(x, bs)
        assert np.allclose(got, want, atol=1e-12), f"block_size={bs}: expected {want[:3]}..., got {got[:3]}..."


def test_last_partial_block_is_kept():
    # The largest value is in the leftover block.
    x = np.array([0.0, 0.0, 0.0, 0.0, 9.0])
    got = blocked_softmax(x, 2)
    want = stable_softmax(x)
    assert len(got) == 5, f"expected 5 outputs, got {len(got)}"
    assert np.allclose(got, want, atol=1e-12), f"expected {want}, got {got}"


def test_max_in_first_block_and_in_last_block():
    for x in (np.array([8.0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 8.0])):
        got = blocked_softmax(x, 3)
        want = stable_softmax(x)
        assert np.allclose(got, want, atol=1e-12), f"expected {want}, got {got}"


def test_large_and_negative_values():
    for x in (np.array([1000.0, 1001.0, 999.0, 1002.0, 998.0]), np.array([-900.0, -901.0, -899.0, -905.0])):
        got = blocked_softmax(x, 2)
        want = stable_softmax(x)
        assert np.all(np.isfinite(got)), f"got non-finite values {got}"
        assert np.allclose(got, want, atol=1e-12), f"expected {want}, got {got}"
        assert abs(got.sum() - 1) < 1e-12, f"outputs must sum to 1, got {got.sum()}"
