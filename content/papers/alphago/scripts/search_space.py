"""Numbers for "The problem": the size of the search space b^d.

The paper's introduction gives b = 35, d = 80 for chess and b = 250, d = 150 for Go.

Run: python3 scripts/search_space.py
"""

import math

for name, b, d in (("chess", 35, 80), ("Go", 250, 150)):
    print(f"{name}: b = {b}, d = {d}, log10(b^d) = {d * math.log10(b):.1f}")

# Even a search that cuts the breadth to 5 moves per position (a policy
# network's job) and the depth to 10 plies (a value network's job) is small.
print(f"b = 5, d = 10: {5**10:,} sequences")
# Nim from 7 stones: the full game tree is tiny, which is why we can check a search by hand.
def count(n):
    return 1 if n == 0 else sum(count(n - k) for k in (1, 2, 3) if k <= n)
print(f"Nim, 7 stones: {count(7)} complete games")

# The Elo statement in the Figure 4 caption: a 230-point gap is a 79% win probability.
print(f"Elo gap 230 -> win probability {1 / (1 + 10 ** (-230 / 400)):.3f}")
