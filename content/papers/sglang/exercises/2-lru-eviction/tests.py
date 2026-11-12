from submission import RadixCache


def cached(cache, seq):
    return cache.peek(seq) == len(seq)


def test_evicts_the_least_recently_used_leaf():
    c = RadixCache(100)
    a, b, d = (1, 2, 3), (4, 5), (6, 7, 8, 9)
    for seq in (a, b, d):
        c.insert(seq)
    c.insert(a)  # touch a again, so the order from oldest is b, d, a
    freed = c.evict(1)
    assert freed == 2, f"the oldest leaf is b with 2 tokens, expected 2 freed, got {freed}"
    assert not cached(c, b), "b is the least recently used leaf and must be gone"
    assert cached(c, a) and cached(c, d), "a and d must stay"


def test_evicts_whole_leaves_until_enough_is_freed():
    c = RadixCache(100)
    a, b, d = (1, 2, 3), (4, 5), (6, 7, 8, 9)
    for seq in (a, b, d):
        c.insert(seq)
    c.insert(a)
    freed = c.evict(3)
    assert freed == 6, f"b (2 tokens) is not enough, so d (4 tokens) goes too; expected 6 freed, got {freed}"
    assert c.size() == 3, f"only a remains, expected size 3, got {c.size()}"
    assert cached(c, a), "a is the most recently used and stays"


def test_returns_tokens_not_nodes():
    c = RadixCache(100)
    c.insert((1, 2, 3, 4, 5))
    freed = c.evict(1)
    assert freed == 5, f"one leaf of 5 tokens: expected 5 tokens freed, got {freed}"
    assert c.size() == 0, f"expected an empty cache, got size {c.size()}"


def test_zero_request_frees_nothing():
    c = RadixCache(100)
    c.insert((1, 2, 3))
    assert c.evict(0) == 0, "asking for 0 tokens frees 0"
    assert c.size() == 3, f"nothing is removed, expected size 3, got {c.size()}"


def test_only_leaves_are_evicted_and_parents_follow():
    c = RadixCache(100)
    c.insert((1, 2, 3, 4))   # time 1
    c.insert((1, 2, 5, 6))   # time 2: splits into (1, 2) with leaves (3, 4) and (5, 6)
    freed = c.evict(3)
    assert freed == 4, f"leaf (3, 4) goes first, then leaf (5, 6); expected 4 freed, got {freed}"
    assert c.size() == 2, f"the shared node (1, 2) must survive: expected size 2, got {c.size()}"
    assert c.peek((1, 2)) == 2, "the shared prefix (1, 2) stays cached"


def test_a_parent_becomes_evictable_once_its_children_are_gone():
    c = RadixCache(100)
    c.insert((1, 2, 3, 4))
    c.insert((1, 2, 5, 6))
    freed = c.evict(5)
    assert freed == 6, f"leaf, leaf, then the parent (1, 2): expected 6 freed, got {freed}"
    assert c.size() == 0, f"expected an empty cache, got size {c.size()}"


def test_size_drops_by_exactly_what_was_freed():
    c = RadixCache(100)
    c.insert((1, 2, 3, 4))
    c.insert((1, 2, 5, 6))
    c.insert((9, 9))
    before = c.size()
    freed = c.evict(3)
    assert before - c.size() == freed, f"size fell by {before - c.size()} but evict reported {freed}: a subtree was dropped"


def test_locked_requests_are_never_evicted():
    c = RadixCache(100)
    a, b = (1, 2, 3, 4), (1, 2, 5, 6)
    c.insert(a)
    c.insert(b)
    c.lock(a)
    freed = c.evict(100)
    assert freed == 2, f"only leaf (5, 6) is free to evict, expected 2 freed, got {freed}"
    assert cached(c, a), "the locked request must stay fully cached"
    assert c.peek(b) == 2, f"b keeps only the shared prefix, expected match 2, got {c.peek(b)}"


def test_unlock_makes_the_request_evictable_again():
    c = RadixCache(100)
    a = (1, 2, 3, 4)
    c.insert(a)
    c.lock(a)
    assert c.evict(10) == 0, "a locked leaf cannot be evicted, so 0 tokens are freed"
    c.unlock(a)
    assert c.evict(10) == 4, "after unlock the leaf can go"


def test_a_lock_taken_before_a_split_still_protects_both_halves():
    c = RadixCache(100)
    a = (1, 2, 3, 4)
    c.insert(a)
    c.lock(a)
    c.insert((1, 2, 9))      # splits the locked edge
    freed = c.evict(100)
    assert freed == 1, f"only the new leaf (9,) is free, expected 1 freed, got {freed}"
    assert cached(c, a), "the locked request survives the split"


def test_least_recently_used_beats_oldest_inserted():
    c = RadixCache(100)
    x, y, z = (1, 1), (2, 2), (3, 3)
    for seq in (x, y, z):
        c.insert(seq)
    c.insert(x)
    c.insert(y)   # use order, oldest first: z, x, y
    c.evict(1)
    assert not cached(c, z), "z was used longest ago and must be the first to go"
    assert cached(c, x) and cached(c, y), "x and y were used more recently"
    c.evict(1)
    assert not cached(c, x) and cached(c, y), "x goes second, y last"


def test_unlock_after_a_split_leaves_nothing_stuck():
    c = RadixCache(100)
    a = (1, 2, 3, 4)
    c.insert(a)
    c.lock(a)
    c.insert((1, 2, 9))      # splits the locked edge
    c.unlock(a)
    freed = c.evict(100)
    assert freed == 5, f"after unlock all 5 tokens can go, expected 5 freed, got {freed}"
    assert c.size() == 0, f"expected an empty cache, got size {c.size()}"
