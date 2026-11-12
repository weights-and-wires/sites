import random

from submission import RadixCache, pick_next, serve


def toy_trace():
    prompts = {g: tuple(range(100 * g, 100 * g + 6)) for g in range(3)}
    reqs = []
    for j in range(3):
        for g in range(3):
            reqs.append(prompts[g] + (1000 + 10 * g + j, 2000 + 10 * g + j))
    return reqs


def test_fcfs_takes_the_oldest():
    c = RadixCache(100)
    c.admit((1, 2, 3))
    waiting = [(9, 9), (1, 2, 3, 4)]
    pos = pick_next(waiting, c, "fcfs")
    assert pos == 0, f"fcfs ignores the cache and picks position 0, got {pos}"


def test_lpm_takes_the_longest_cached_prefix():
    c = RadixCache(100)
    c.admit((1, 2, 3, 4))
    waiting = [(9, 9, 9), (1, 2, 8, 8), (1, 2, 3, 7), (5, 5)]
    pos = pick_next(waiting, c, "lpm")
    assert pos == 2, f"matches are 0, 2, 3, 0, so position 2 wins, got {pos}"


def test_lpm_breaks_ties_by_arrival():
    c = RadixCache(100)
    c.admit((1, 2, 3))
    waiting = [(9, 9), (1, 2, 5), (1, 2, 6), (1, 2, 7)]
    pos = pick_next(waiting, c, "lpm")
    assert pos == 1, f"three requests match 2 tokens; the earliest is position 1, got {pos}"
    pos = pick_next([(8, 8), (9, 9)], c, "lpm")
    assert pos == 0, f"with no matches the oldest wins, expected 0, got {pos}"


def test_pick_next_does_not_change_the_cache():
    c = RadixCache(100)
    c.admit((1, 2, 3))
    before = (c.size(), c.clock)
    pick_next([(1, 2, 5), (1, 9)], c, "lpm")
    assert (c.size(), c.clock) == before, f"choosing must not touch the cache: expected {before}, got {(c.size(), c.clock)}"


def test_serve_counts_hits_by_hand():
    reqs = [(1, 2, 3), (1, 2, 4), (1, 2, 3)]
    hits, total, order = serve(reqs, 100, "fcfs")
    assert (hits, total) == (5, 9), f"hits 0 + 2 + 3 of 9 tokens, expected (5, 9), got {(hits, total)}"
    assert order == [0, 1, 2], f"fcfs serves in arrival order, got {order}"
    hits, total, order = serve(reqs, 100, "lpm")
    assert order == [0, 2, 1], f"after request 0, request 2 matches 3 tokens and goes next; got {order}"
    assert (hits, total) == (5, 9), f"hits 0 + 3 + 2 of 9 tokens, expected (5, 9), got {(hits, total)}"


def test_serve_serves_every_request_once():
    reqs = toy_trace()
    for policy in ("fcfs", "lpm"):
        hits, total, order = serve(reqs, 20, policy)
        assert sorted(order) == list(range(len(reqs))), f"{policy}: every request is served once, got order {order}"
        assert total == sum(len(r) for r in reqs), f"{policy}: total counts every token, expected {sum(len(r) for r in reqs)}, got {total}"


def test_thrashing_trace_fcfs_vs_lpm():
    reqs = toy_trace()
    hits, total, order = serve(reqs, 20, "fcfs")
    assert (hits, total) == (0, 72), f"interleaved prompts evict each other under fcfs: expected (0, 72), got {(hits, total)}"
    hits, total, order = serve(reqs, 20, "lpm")
    assert (hits, total) == (36, 72), f"lpm groups requests by prompt: expected (36, 72), got {(hits, total)}"
    assert order == [0, 3, 6, 1, 4, 7, 2, 5, 8], f"expected the three requests of each prompt together, got {order}"


def test_a_big_cache_makes_the_order_irrelevant():
    reqs = toy_trace()
    a = serve(reqs, 1000, "fcfs")[0]
    b = serve(reqs, 1000, "lpm")[0]
    assert a == b == 36, f"with room for everything both orders hit 36 tokens, got fcfs {a} and lpm {b}"


def test_lpm_matches_dfs_order_for_one_batch():
    rng = random.Random(3)
    for trial in range(40):
        base = [tuple(rng.randrange(3) for _ in range(rng.randrange(2, 5))) for _ in range(4)]
        reqs = [rng.choice(base) + tuple(rng.randrange(4) for _ in range(rng.randrange(1, 5))) for _ in range(12)]
        cap = max(map(len, reqs)) + rng.randrange(0, 12)
        lpm = serve(reqs, cap, "lpm")[0]
        dfs = serve(sorted(reqs), cap, "fcfs")[0]
        fcfs = serve(reqs, cap, "fcfs")[0]
        assert lpm == dfs, f"trial {trial}: longest-prefix-first should equal DFS (sorted) order, got {lpm} and {dfs}"
        assert lpm >= fcfs, f"trial {trial}: lpm should not lose to fcfs, got {lpm} and {fcfs}"
