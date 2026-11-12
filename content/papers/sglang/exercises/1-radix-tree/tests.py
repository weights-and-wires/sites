import random

from submission import RadixTree


def edges(tree):
    """Sorted list of (edge tokens) over all non-root nodes, to compare structure."""
    return sorted(n.edge for n in tree._nodes() if n is not tree.root)


def test_first_insert_is_one_node():
    t = RadixTree()
    new = t.insert([1, 2, 3, 4])
    assert new == 4, f"all 4 tokens are new, expected 4, got {new}"
    assert t.num_nodes() == 1, f"one sequence is one edge, expected 1 node, got {t.num_nodes()}"
    assert t.num_tokens() == 4, f"expected 4 stored tokens, got {t.num_tokens()}"


def test_divergence_splits_an_edge():
    t = RadixTree()
    t.insert([1, 2, 3, 4])
    new = t.insert([1, 2, 5])
    assert new == 1, f"only token 5 is new, expected 1, got {new}"
    assert edges(t) == [(1, 2), (3, 4), (5,)], f"expected edges (1, 2), (3, 4), (5,), got {edges(t)}"
    assert t.num_tokens() == 5, f"expected 5 stored tokens, got {t.num_tokens()}"


def test_no_common_prefix_adds_a_root_child():
    t = RadixTree()
    t.insert([1, 2, 3])
    new = t.insert([7, 8])
    assert new == 2, f"expected 2 new tokens, got {new}"
    assert sorted(t.root.children) == [1, 7], f"root children keyed by first token: expected [1, 7], got {sorted(t.root.children)}"
    assert t.num_nodes() == 2, f"expected 2 nodes, got {t.num_nodes()}"


def test_inserting_a_cached_sequence_changes_nothing():
    t = RadixTree()
    t.insert([1, 2, 3, 4])
    new = t.insert([1, 2, 3, 4])
    assert new == 0, f"nothing is new, expected 0, got {new}"
    assert edges(t) == [(1, 2, 3, 4)], f"expected one edge, got {edges(t)}"


def test_inserting_a_prefix_splits_at_its_end():
    t = RadixTree()
    t.insert([1, 2, 3, 4])
    new = t.insert([1, 2])
    assert new == 0, f"a stored prefix adds no tokens, expected 0, got {new}"
    assert edges(t) == [(1, 2), (3, 4)], f"the prefix ends at a node: expected (1, 2) and (3, 4), got {edges(t)}"
    assert t.num_tokens() == 4, f"expected 4 stored tokens, got {t.num_tokens()}"


def test_extending_a_stored_sequence():
    t = RadixTree()
    t.insert([1, 2])
    new = t.insert([1, 2, 3, 4])
    assert new == 2, f"expected 2 new tokens, got {new}"
    assert edges(t) == [(1, 2), (3, 4)], f"expected edges (1, 2) and (3, 4), got {edges(t)}"


def test_parent_links_after_a_split():
    t = RadixTree()
    t.insert([1, 2, 3, 4])
    t.insert([1, 2, 5])
    top = t.root.children[1]
    assert top.parent is t.root, "the shared node hangs under the root"
    for first, child in top.children.items():
        assert child.parent is top, f"child {first} must point at the shared node after the split"
        assert child.edge[0] == first, f"children are keyed by the first token of their edge; got key {first} for edge {child.edge}"


def test_match_stops_inside_an_edge():
    t = RadixTree()
    t.insert([1, 2, 3, 4])
    assert t.match([1, 2, 3, 9]) == 3, f"expected 3, got {t.match([1, 2, 3, 9])}"
    assert t.match([1, 2]) == 2, f"a prefix that ends inside an edge counts: expected 2, got {t.match([1, 2])}"
    assert t.match([1, 2, 3, 4, 5, 6]) == 4, f"expected 4, got {t.match([1, 2, 3, 4, 5, 6])}"
    assert t.match([9, 1, 2]) == 0, f"no match on the first token: expected 0, got {t.match([9, 1, 2])}"
    assert t.match([]) == 0, "the empty request matches nothing"


def test_match_does_not_change_the_tree():
    t = RadixTree()
    t.insert([1, 2, 3, 4])
    t.insert([1, 2, 5])
    before = (t.num_nodes(), t.num_tokens())
    t.match([1, 2, 3, 9, 9])
    t.match([1, 2, 6])
    assert (t.num_nodes(), t.num_tokens()) == before, f"match must not insert: expected {before}, got {(t.num_nodes(), t.num_tokens())}"


def test_match_across_a_split():
    t = RadixTree()
    t.insert([1, 2, 3, 4])
    t.insert([1, 2, 5, 6])
    assert t.match([1, 2, 5, 6, 7]) == 4, f"expected 4, got {t.match([1, 2, 5, 6, 7])}"
    assert t.match([1, 2, 5, 9]) == 3, f"expected 3, got {t.match([1, 2, 5, 9])}"
    assert t.match([1, 2, 9]) == 2, f"expected 2, got {t.match([1, 2, 9])}"


def test_insert_leaves_the_caller_sequence_alone():
    t = RadixTree()
    seq = [1, 2, 3]
    t.insert(seq)
    t.insert([1, 2, 4])
    assert seq == [1, 2, 3], f"insert must not modify its argument, got {seq}"


def test_against_brute_force():
    rng = random.Random(7)
    for trial in range(60):
        t = RadixTree()
        stored = []
        for _ in range(rng.randrange(1, 9)):
            seq = tuple(rng.randrange(3) for _ in range(rng.randrange(1, 7)))
            prefixes = {s[:k] for s in stored for k in range(len(s) + 1)}
            want_new = sum(1 for k in range(1, len(seq) + 1) if seq[:k] not in prefixes)
            got_new = t.insert(seq)
            assert got_new == want_new, f"trial {trial}: inserting {seq} into {stored}, expected {want_new} new tokens, got {got_new}"
            stored.append(seq)
        prefixes = {s[:k] for s in stored for k in range(len(s) + 1)}
        assert t.num_tokens() == len(prefixes) - 1, f"trial {trial}: expected {len(prefixes) - 1} stored tokens, got {t.num_tokens()}"
        for _ in range(10):
            q = tuple(rng.randrange(3) for _ in range(rng.randrange(0, 8)))
            want = max(k for k in range(len(q) + 1) if q[:k] in prefixes)
            assert t.match(q) == want, f"trial {trial}: match {q} against {stored}, expected {want}, got {t.match(q)}"
