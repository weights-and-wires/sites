"""Worked numbers for "A cache of sinks plus a rolling window".

Run from the paper folder: python3 scripts/cache_indices.py

Part 1 reproduces the cache of Figure 4 (Section 3.2). Part 2 is this page's
own sweep: how many slots a 4 + 8 cache holds, and which tokens, as the text grows.
"""


def kept_indices(n, sinks, window):
    """Token indices in the cache after n tokens: the first `sinks`, then the last `window`."""
    if n <= sinks + window:
        return list(range(n))
    return list(range(sinks)) + list(range(n - window, n))


# Part 1: Figure 4. Decode token 9 with 4 sinks and the 3 most recent tokens cached.
sinks, recent = 4, 3
n_seen = 9  # tokens 0..8 are already generated
kept = kept_indices(n_seen, sinks, recent)
evicted = [i for i in range(n_seen) if i not in kept]
print("Figure 4: cache before token 9:", kept, "| evicted:", evicted)
print("Figure 4: positions used while decoding token 9:", list(range(len(kept) + 1)))

# Part 2: 4 sinks and a window of 8 (12 slots).
print("\nsinks=4, window=8")
for n in (5, 12, 13, 20, 100):
    k = kept_indices(n, 4, 8)
    print(f"n={n:3d}: slots={len(k):2d} kept={k} evicted={n - len(k)}")
