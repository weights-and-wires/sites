"""Which solutions to send to labelers (Section 2.4 and Section 4.2).

A pool of 12 solutions scored by a selector PRM. Each has a final-answer flag.
Pick k = 5 with the 80 percent / 20 percent rule of Section 4.2.

Run: python3 scripts/active_learning.py
"""

# (selector score, final answer is wrong)
pool = [
    (0.91, True), (0.88, False), (0.86, True), (0.80, True), (0.77, False), (0.74, True),
    (0.70, True), (0.62, False), (0.55, True), (0.40, True), (0.31, False), (0.12, True),
]
k, frac_wrong = 5, 0.8
n_wrong = round(frac_wrong * k)
n_rest = k - n_wrong

wrong = sorted((i for i, (_, w) in enumerate(pool) if w), key=lambda i: -pool[i][0])
chosen_wrong = wrong[:n_wrong]
rest = sorted((i for i in range(len(pool)) if i not in chosen_wrong), key=lambda i: -pool[i][0])
chosen_rest = rest[:n_rest]

print(f"k={k}: {n_wrong} convincing wrong-answer solutions, {n_rest} most convincing of the rest")
print(f"wrong-answer picks (index, score): {[(i, pool[i][0]) for i in chosen_wrong]}")
print(f"remaining pick (index, score):     {[(i, pool[i][0]) for i in chosen_rest]}")
print(f"labeled: {sorted(chosen_wrong + chosen_rest)}")

uniform_wrong = sum(w for _, w in pool) / len(pool)
chosen = chosen_wrong + chosen_rest
print(f"wrong-answer share: pool {uniform_wrong:.3f}, labeled set {sum(pool[i][1] for i in chosen) / k:.3f}")
