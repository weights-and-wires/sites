"""A compute-optimal strategy by difficulty bin (Section 3.1, Equation 1; Section 3.2).

Part 1 bins 200 toy questions into five quantiles of an estimated pass@1.
Part 2 picks the best strategy per bin from a toy accuracy table.

The accuracy table is invented for this page. It only mimics the shape the
paper reports in Section 5.3: beam search helps on middle bins, best-of-N is
safer on easy bins, and nothing helps much on the hardest bin.

Run: python3 scripts/compute_optimal_bins.py
"""

import numpy as np

rng = np.random.default_rng(0)
N_QUESTIONS, N_BINS, N_SAMPLES = 200, 5, 2048

# Part 1: true pass@1 per question, then the 2048-sample estimate the paper uses.
true_p = rng.beta(0.6, 1.2, size=N_QUESTIONS)
pass1 = rng.binomial(N_SAMPLES, true_p) / N_SAMPLES
order = np.argsort(-pass1, kind="stable")                  # easiest question first
bins = np.empty(N_QUESTIONS, dtype=int)
bins[order] = (np.arange(N_QUESTIONS) * N_BINS) // N_QUESTIONS   # 0 = easiest bin
sizes = np.bincount(bins, minlength=N_BINS)
print(f"{N_QUESTIONS} questions, pass@1 estimated from {N_SAMPLES} samples each")
for b in range(N_BINS):
    mask = bins == b
    print(f"  bin {b + 1}: {sizes[b]} questions, mean pass@1 {pass1[mask].mean():.3f}")

# Part 2: toy accuracy of each strategy in each bin, at one fixed budget.
STRATEGIES = ["best-of-N weighted", "beam search M=4", "revisions"]
ACC = np.array([
    # bin 1  bin 2  bin 3  bin 4  bin 5
    [0.92, 0.80, 0.55, 0.30, 0.06],    # best-of-N weighted
    [0.88, 0.78, 0.62, 0.38, 0.07],    # beam search M=4
    [0.95, 0.79, 0.50, 0.25, 0.04],    # revisions
])
print()
print("toy accuracy table (rows are strategies, columns are bins 1 to 5)")
for name, row in zip(STRATEGIES, ACC):
    print(f"  {name:20s}" + "  ".join(f"{a:.2f}" for a in row))

best_idx = ACC.argmax(axis=0)
print()
print("best strategy per bin: " + ", ".join(
    f"bin {b + 1}={STRATEGIES[i]}" for b, i in enumerate(best_idx)))

weights = sizes / sizes.sum()
fixed = ACC @ weights
optimal = float((ACC.max(axis=0) * weights).sum())
print()
for name, acc in zip(STRATEGIES, fixed):
    print(f"fixed strategy {name:20s} overall accuracy {acc:.4f}")
print(f"compute-optimal (best per bin)    overall accuracy {optimal:.4f}")
print(f"gain over the best fixed strategy: {optimal - fixed.max():.4f}")
