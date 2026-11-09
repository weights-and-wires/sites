"""Count memory accesses per element for the three softmax algorithms and the top-k fusion.

Every read or write of the input or output vector goes through a counting wrapper.
Run: python3 scripts/access_counts.py
"""

import math


class Mem:
    def __init__(self, data=None, size=0):
        self.data = list(data) if data is not None else [0.0] * size
        self.loads = 0
        self.stores = 0

    def __getitem__(self, i):
        self.loads += 1
        return self.data[i]

    def __setitem__(self, i, v):
        self.stores += 1
        self.data[i] = v


def naive(x, y, V):  # Algorithm 1
    d = 0.0
    for j in range(V):
        d += math.exp(x[j])
    for i in range(V):
        y[i] = math.exp(x[i]) / d


def safe(x, y, V):  # Algorithm 2
    m = -math.inf
    for k in range(V):
        m = max(m, x[k])
    d = 0.0
    for j in range(V):
        d += math.exp(x[j] - m)
    for i in range(V):
        y[i] = math.exp(x[i] - m) / d


def online(x, y, V):  # Algorithm 3
    m, d = -math.inf, 0.0
    for j in range(V):
        xj = x[j]
        m_new = max(m, xj)
        d = d * math.exp(m - m_new) + math.exp(xj - m_new)
        m = m_new
    for i in range(V):
        y[i] = math.exp(x[i] - m) / d


V = 1000
data = [math.sin(i) * 3 for i in range(V)]
counts = {}
for fn in (naive, safe, online):
    x, y = Mem(data), Mem(size=V)
    fn(x, y, V)
    counts[fn.__name__] = (x.loads + y.loads + x.stores + y.stores) / V
    print(f"{fn.__name__:<7}: {x.loads / V:.0f} loads of x + {y.stores / V:.0f} store of y = {counts[fn.__name__]:.0f} accesses per element")

print(f"safe / online = {counts['safe'] / counts['online']:.2f}")

# Softmax followed by top-k: top-k loads every y once.
print()
print("softmax then top-k, accesses per element:")
print(f"  safe + top-k run separately  : {counts['safe']:.0f} + 1 = {counts['safe'] + 1:.0f}")
print(f"  online + top-k run separately: {counts['online']:.0f} + 1 = {counts['online'] + 1:.0f}")
print("  online fused with top-k      : 1 (one load of x; only K values and indices are written)")
print(f"  ratio, safe separate / fused : {(counts['safe'] + 1) / 1:.0f}x")
