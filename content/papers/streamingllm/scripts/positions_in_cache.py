"""Worked numbers for "Positions come from the cache, not from the text".

Run from the paper folder: python3 scripts/positions_in_cache.py

Cache of Figure 4: tokens 0 1 2 3 6 7 8 are cached and token 9 is decoded.
Keys are stored before the rotary transform (Section 3.2), so the rotation
uses positions that this script chooses at read time.
"""
import numpy as np

kept = [0, 1, 2, 3, 6, 7, 8, 9]  # token 9 is the query, last
orig = np.array(kept)
cache = np.arange(len(kept))
print("original positions:", orig.tolist())
print("cache positions:   ", cache.tolist())
print("distance from token 9 to the sink 0: original", orig[-1] - orig[0], "| cache", cache[-1] - cache[0])
print("distance from token 9 to token 6:    original", orig[-1] - orig[4], "| cache", cache[-1] - cache[4])
print("gap between token 3 and token 6:     original", orig[4] - orig[3], "| cache", cache[4] - cache[3])

# One rotary pair with theta = 1 (the fastest pair, j = 0). For q = k = [1, 0],
# the score is cos(distance * theta).
for name, dist in [("token 9 to sink 0, original distance", 9), ("token 9 to sink 0, cache distance", 7)]:
    print(f"score for {name} {dist}: {np.cos(dist):.4f}")
