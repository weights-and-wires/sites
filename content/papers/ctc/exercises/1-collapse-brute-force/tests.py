import itertools

import numpy as np

from submission import brute_force_ctc, collapse

Y = np.array([[0.4, 0.5, 0.1], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])  # columns: blank, a, b


def test_collapse_paper_example():
    # The paper's example: B(a-ab-) = B(-aa--abb) = aab, with a=1, b=2, blank=0.
    p1 = (1, 0, 1, 2, 0)
    p2 = (0, 1, 1, 0, 0, 1, 2, 2)
    assert collapse(p1) == (1, 1, 2), f"B(a-ab-): expected (1, 1, 2), got {collapse(p1)}"
    assert collapse(p2) == (1, 1, 2), f"B(-aa--abb): expected (1, 1, 2), got {collapse(p2)}"


def test_collapse_merges_repeats_without_blank():
    assert collapse((1, 1, 1)) == (1,), f"B(aaa): expected (1,), got {collapse((1, 1, 1))}"
    assert collapse((1, 1, 2, 2)) == (1, 2), f"B(aabb): expected (1, 2), got {collapse((1, 1, 2, 2))}"


def test_blank_separates_repeats():
    assert collapse((1, 0, 1)) == (1, 1), f"B(a-a): expected (1, 1), got {collapse((1, 0, 1))}"
    assert collapse((1, 0, 0, 1)) == (1, 1), f"B(a--a): expected (1, 1), got {collapse((1, 0, 0, 1))}"


def test_all_blank_and_empty():
    assert collapse((0, 0, 0)) == (), f"B(---): expected (), got {collapse((0, 0, 0))}"
    assert collapse(()) == (), f"B(empty path): expected (), got {collapse(())}"


def test_nonzero_blank_index():
    got = collapse((2, 0, 0, 2, 1), blank=2)
    assert got == (0, 1), f"blank=2: expected (0, 1), got {got}"


def test_brute_force_page_example():
    for label, want in [((1, 2), 0.273), ((1,), 0.351), ((1, 1), 0.03), ((), 0.06)]:
        got = brute_force_ctc(Y, label)
        assert abs(got - want) < 1e-9, f"p({label}|x): expected {want}, got {got}"


def test_repeated_label_needs_a_blank():
    # With T=2 there is no room for a-a, so p(aa|x) is exactly 0.
    y = np.array([[0.2, 0.8], [0.2, 0.8]])
    got = brute_force_ctc(y, (1, 1))
    assert got == 0.0, f"p(aa|x) with T=2: expected 0.0, got {got}"
    got = brute_force_ctc(y, (1,))
    assert abs(got - 0.96) < 1e-9, f"p(a|x) with T=2: expected 0.96 (aa, a-, -a), got {got}"


def test_labellings_sum_to_one():
    rng = np.random.default_rng(0)
    y = rng.random((4, 3))
    y /= y.sum(axis=1, keepdims=True)
    labellings = {()}
    for n in range(1, 5):
        labellings |= set(itertools.product([1, 2], repeat=n))
    total = sum(brute_force_ctc(y, l) for l in labellings)
    assert abs(total - 1.0) < 1e-9, f"every path maps to some labelling, so the sum must be 1, got {total}"
