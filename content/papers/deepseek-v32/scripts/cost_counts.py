"""Counts of query-key pairs for dense and sparse attention, and the
training-token arithmetic from Section 2.1.1.

Section 2.3 states that DSA lowers the core attention cost from O(L^2) to
O(L k) while the indexer stays O(L^2). This script counts pairs only. The
paper gives no indexer width or head count, so it cannot turn pairs into
FLOPs or dollars; Figure 3 reports measured cost per million tokens as a
chart, not as numbers in the text.

Run: python3 scripts/cost_counts.py
"""


def dense_pairs(L):
    return L * (L + 1) // 2


def sparse_pairs(L, k):
    return sum(min(t + 1, k) for t in range(L))


k = 2048
for L in (2048, 8192, 32768, 131072):
    d, s = dense_pairs(L), sparse_pairs(L, k)
    print(f"L={L:>6}: dense {d:>14,}  DSA main {s:>13,}  indexer {d:>14,}  main ratio {d / s:.2f}")

L = 131072
print("128K = 131072 tokens; L k =", L * k)

warm = 1000 * 16 * 131072
sparse = 15000 * 480 * 131072
print("warm-up tokens  =", warm, f"({warm / 1e9:.1f}B)")
print("sparse tokens   =", sparse, f"({sparse / 1e9:.1f}B)")
print("sparse / warm-up =", round(sparse / warm, 1))

print("\nself-check toy: L=6, k=3")
print("dense pairs =", dense_pairs(6), " DSA main pairs =", sparse_pairs(6, 3))
