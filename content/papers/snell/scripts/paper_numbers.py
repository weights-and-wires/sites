"""Arithmetic on the paper's reported setup, so the page quotes checked numbers.

Sources: Section 3.2 (difficulty bins), Section 4 (data split), Section 5.2
(beam search, lookahead cost), Section 5.3 and Section 6.2 (4x claims),
Section 6.1 (revision flips), Section 7 (the R values).

Run: python3 scripts/paper_numbers.py
"""

import math

print("Difficulty bins (Sections 3.2 and 4)")
test_questions, bins = 500, 5
print(f"  {test_questions} test questions in {bins} quantile bins: {test_questions // bins} per bin")

print()
print("Beam width M = sqrt(N) for generation budgets N (Section 5.3)")
for n in (4, 16, 64, 256):
    m = int(math.isqrt(n))
    print(f"  N={n:3d}: M={m:2d}, beams kept per round N/M={n // m:2d}")

print()
print("Cost of lookahead search, N x (k + 1) samples (Section 5.3)")
for n, k in ((16, 0), (16, 1), (16, 3), (256, 3)):
    print(f"  N={n:3d}, k={k}: {n * (k + 1)} samples")

print()
print("The 4x claims (Sections 5.3 and 6.2)")
for name, small, large in (("PRM search", 16, 64), ("revisions", 64, 256)):
    print(f"  {name}: {large} generations vs {small} generations is {large / small:.0f}x")

print()
print("Sequential and parallel split with budget 256 (Section 6.2)")
root = math.isqrt(256)
print(f"  sqrt(256) = {root}: {root} parallel chains of {root} revisions each = {root * root} generations")

print()
print("The R values (Section 7), D_inference / D_pretrain, and its inverse")
for r in (0.16, 0.79, 22):
    print(f"  R={r:5.2f}: D_pretrain / D_inference = {1 / r:.3f}")
