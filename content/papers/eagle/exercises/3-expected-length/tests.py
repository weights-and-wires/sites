import numpy as np
from submission import expected_length


def test_no_draft_gives_one_token():
    got = expected_length([], [])
    assert abs(got - 1.0) < 1e-12, f"expected 1.0, got {got}"


def test_chain_matches_the_geometric_sum():
    for alpha, d in ((0.5, 3), (0.7, 5), (0.9, 8)):
        parents = list(range(-1, d - 1))
        want = sum(alpha ** k for k in range(d + 1))
        got = expected_length(parents, [alpha] * d)
        assert abs(got - want) < 1e-9, f"alpha={alpha}, d={d}: expected {want}, got {got}"


def test_two_siblings_at_the_root():
    # 1 + 0.6 + (1 - 0.6) * 0.5
    got = expected_length([-1, -1], [0.6, 0.5])
    assert abs(got - 1.8) < 1e-9, f"expected 1.8, got {got}"


def test_child_only_counts_when_its_parent_is_accepted():
    # a (0.5) -> b (0.5). 1 + 0.5 + 0.25
    got = expected_length([-1, 0], [0.5, 0.5])
    assert abs(got - 1.75) < 1e-9, f"expected 1.75, got {got}"


def test_sibling_branch_continues_from_the_sibling():
    # root children: 0 (0.5) and 1 (0.4). Node 2 is a child of node 1 with accept 0.5.
    # P(0) = 0.5. P(1) = 0.5 * 0.4 = 0.2. P(2) = 0.2 * 0.5 = 0.1.
    got = expected_length([-1, -1, 1], [0.5, 0.4, 0.5])
    assert abs(got - 1.8) < 1e-9, f"expected 1 + 0.5 + 0.2 + 0.1 = 1.8, got {got}"


def test_third_sibling_needs_both_earlier_siblings_rejected():
    # 1 + 0.5 + 0.5 * 0.5 + 0.5 * 0.5 * 0.5
    got = expected_length([-1, -1, -1], [0.5, 0.5, 0.5])
    assert abs(got - 1.875) < 1e-9, f"expected 1.875, got {got}"


def test_the_ten_node_tree_from_the_page():
    parents, accept = [], []
    spine = -1
    for _ in range(5):
        parents += [spine, spine]
        accept += [0.7, 0.5]
        spine = len(parents) - 2
    got = expected_length(parents, accept)
    assert abs(got - 3.3571) < 1e-3, f"expected 3.3571, got {got:.4f}"


def test_adding_a_sibling_never_lowers_the_length():
    chain = expected_length([-1, 0, 1], [0.7] * 3)
    tree = expected_length([-1, 0, 1, -1], [0.7, 0.7, 0.7, 0.5])
    assert tree > chain, f"expected the tree ({tree:.4f}) to beat the chain ({chain:.4f})"
    want = chain + 0.3 * 0.5
    assert abs(tree - want) < 1e-9, f"the sibling adds P(first rejected) * 0.5 = 0.15, expected {want}, got {tree}"
