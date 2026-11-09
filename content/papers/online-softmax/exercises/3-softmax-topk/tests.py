import math
import random

from submission import online_softmax_topk


def reference(x, k):
    m = max(x)
    d = sum(math.exp(v - m) for v in x)
    order = sorted(range(len(x)), key=lambda i: (-x[i], i))[:k]
    return [math.exp(x[i] - m) / d for i in order], order


def test_small_example():
    x = [2.0, 1.0, 4.0, 3.0]
    v, z = online_softmax_topk(iter(x), 2)
    assert z == [2, 3], f"expected indices [2, 3], got {z}"
    assert abs(v[0] - 0.6439) < 1e-4 and abs(v[1] - 0.2369) < 1e-4, f"expected [0.6439, 0.2369], got {v}"


def test_probabilities_use_the_full_normalizer():
    # If d were summed over the kept elements only, the values would add to 1.
    x = [0.0, 0.0, 0.0, 0.0, 0.0]
    v, _ = online_softmax_topk(iter(x), 2)
    assert all(abs(a - 0.2) < 1e-12 for a in v), f"expected [0.2, 0.2], got {v}"


def test_matches_reference_on_random_data():
    random.seed(0)
    for n, k in [(50, 5), (200, 1), (30, 10), (8, 8)]:
        x = [random.gauss(0, 6) for _ in range(n)]
        want_v, want_z = reference(x, k)
        v, z = online_softmax_topk(iter(x), k)
        assert z == want_z, f"n={n}, k={k}: expected indices {want_z}, got {z}"
        assert all(abs(a - b) < 1e-12 for a, b in zip(v, want_v)), f"n={n}, k={k}: expected {want_v}, got {v}"


def test_descending_order():
    random.seed(3)
    x = [random.gauss(0, 2) for _ in range(40)]
    v, _ = online_softmax_topk(iter(x), 6)
    assert v == sorted(v, reverse=True), f"values must be in descending order, got {v}"


def test_ascending_input_rescales_the_normalizer():
    x = [float(i) for i in range(20)]
    want_v, want_z = reference(x, 3)
    v, z = online_softmax_topk(iter(x), 3)
    assert z == want_z, f"expected indices {want_z}, got {z}"
    assert all(abs(a - b) < 1e-12 for a, b in zip(v, want_v)), f"expected {want_v}, got {v}"


def test_ties_keep_the_earlier_index_first():
    v, z = online_softmax_topk(iter([1.0, 3.0, 3.0, 2.0]), 2)
    assert z == [1, 2], f"expected [1, 2] on a tie, got {z}"
    v, z = online_softmax_topk(iter([3.0, 3.0, 3.0]), 2)
    assert z == [0, 1], f"expected [0, 1] on a three-way tie, got {z}"


def test_large_values_stay_finite():
    x = [1000.0, 1003.0, 999.0, 1001.0]
    v, z = online_softmax_topk(iter(x), 2)
    want_v, want_z = reference(x, 2)
    assert z == want_z, f"expected indices {want_z}, got {z}"
    assert all(math.isfinite(a) for a in v), f"expected finite values, got {v}"
    assert all(abs(a - b) < 1e-12 for a, b in zip(v, want_v)), f"expected {want_v}, got {v}"


def test_reads_the_input_once():
    random.seed(5)
    x = [random.gauss(0, 3) for _ in range(30)]
    want_v, want_z = reference(x, 4)
    v, z = online_softmax_topk(iter(x), 4)  # one-shot iterator
    assert z == want_z, f"expected indices {want_z}, got {z}"
    assert all(abs(a - b) < 1e-12 for a, b in zip(v, want_v)), f"expected {want_v}, got {v}"
