"""Compute the numbers for walltime speedup and operations (Section 3.3-3.5, Theorems 3.8 and 3.11).

Run: python3 scripts/speedup_example.py
"""


def expected_tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha, gamma, c):
    """Theorem 3.8."""
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


def ops_factor(alpha, gamma, c_hat):
    """Theorem 3.11."""
    return (gamma * c_hat + gamma + 1) / expected_tokens(alpha, gamma)


def best_gamma(alpha, c, max_gamma=40):
    return max(range(1, max_gamma + 1), key=lambda g: speedup(alpha, g, c))


print("Worked example: alpha = 0.8, gamma = 5, c = 0.05")
print("tokens per iteration     :", round(expected_tokens(0.8, 5), 4))
print("cost per iteration (T=1) :", round(5 * 0.05 + 1, 4))
print("speedup                  :", round(speedup(0.8, 5, 0.05), 4))
print("same with c = 0          :", round(speedup(0.8, 5, 0.0), 4))
print()
print("Table 1 (c = c_hat = 0): alpha, gamma, ops, speed")
for a, g in [(0.6, 2), (0.7, 3), (0.8, 2), (0.8, 5), (0.9, 2), (0.9, 10)]:
    print(f"  {a} {g:>2}  {ops_factor(a, g, 0):.2f}x  {speedup(a, g, 0):.2f}x")
print()
print("Table 4 expected factor, rounded to one decimal: task row, alpha, c, gamma, EXP")
rows = [
    ("EnDe small t=0", 0.75, 0.02, 7, 3.2), ("EnDe base t=0", 0.8, 0.04, 7, 3.3), ("EnDe large t=0", 0.82, 0.11, 7, 2.5),
    ("EnDe small t=1", 0.62, 0.02, 7, 2.3), ("EnDe base t=1", 0.68, 0.04, 5, 2.4), ("EnDe large t=1", 0.71, 0.11, 3, 2.0),
    ("CNNDM small t=0", 0.65, 0.02, 5, 2.4), ("CNNDM base t=0", 0.73, 0.04, 5, 2.6), ("CNNDM large t=0", 0.74, 0.11, 3, 2.0),
    ("CNNDM small t=1", 0.53, 0.02, 5, 1.9), ("CNNDM base t=1", 0.55, 0.04, 3, 1.8), ("CNNDM large t=1", 0.56, 0.11, 3, 1.6),
]
mismatch = 0
for name, a, c, g, exp in rows:
    got = speedup(a, g, c)
    ok = round(got, 1) == exp
    mismatch += not ok
    assert abs(got - exp) <= 0.1, (name, got, exp)
    print(f"  {name:<16} {a:<5} {c:<5} {g}  computed {got:.3f}  paper {exp}  {'same after rounding' if ok else 'within 0.1'}")
print("rows that differ from the paper by more than 0.1: 0; rows that round differently:", mismatch)
print("(The paper's alpha and c values are themselves rounded, so a 0.1 gap is expected.)")
print()
print("Corollary 3.9: alpha = 0.3, c = 0.2 (alpha > c)")
print("gamma = 1 speedup        :", round(speedup(0.3, 1, 0.2), 4), " (1 + alpha) / (1 + c) =", round(1.3 / 1.2, 4))
print("gamma = 2 speedup        :", round(speedup(0.3, 2, 0.2), 4))
print("alpha = 0.2, c = 0.3 (alpha < c), gamma = 1:", round(speedup(0.2, 1, 0.3), 4), "(slower than baseline)")
print()
print("Optimal gamma, rows alpha, columns c = 0.01, 0.02, 0.05, 0.1")
for a in (0.5, 0.6, 0.7, 0.8, 0.9):
    print(f"  alpha {a}:", [best_gamma(a, c) for c in (0.01, 0.02, 0.05, 0.1)])
print()
print("Speedup at the optimal gamma, same grid")
for a in (0.5, 0.6, 0.7, 0.8, 0.9):
    print(f"  alpha {a}:", [round(speedup(a, best_gamma(a, c), c), 2) for c in (0.01, 0.02, 0.05, 0.1)])
