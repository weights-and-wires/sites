# Word error rate with an alignment (substitutions, deletions, insertions).

def align(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1][j - 1] + (r[i - 1] != h[j - 1])
            d[i][j] = min(sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("ok" if r[i - 1] == h[j - 1] else "sub", r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", r[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", h[j - 1]))
            j -= 1
    return ops[::-1], len(r)


pairs = [
    ("call zero two zero now", "call zero two oh now"),
    ("the fourteenth of march", "the fourteenth march"),
    ("X K four four seven one", "x k for four seven one"),
    ("one thousand two hundred and forty nine dollars", "one thousand two hundred forty nine dollars please"),
]
errors = words = 0
table = []
for ref, hyp in pairs:
    ops, n = align(ref, hyp)
    e = sum(op[0] != "ok" for op in ops)
    errors, words = errors + e, words + n
    print(f"{e}/{n} = {100 * e / n:.1f}%  ref: {ref!r}  hyp: {hyp!r}")
    table.append([ref, hyp, e, n])
# The post pools errors over all words (90 in 4,377), not the mean of per-sentence rates.
print(f"pooled: {errors}/{words} = {100 * errors / words:.2f}%")
print(f"mean of per-sentence rates: {100 * sum(t[2] / t[3] for t in table) / len(table):.2f}%")
result = table
