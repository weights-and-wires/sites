import numpy as np


def difficulty_bins(pass1, n_bins=5):
    """Return bin indices, 0 for the easiest (highest pass@1) up to n_bins - 1."""
    pass1 = np.asarray(pass1, dtype=float)
    n = len(pass1)
    order = np.argsort(-pass1, kind="stable")
    bins = np.empty(n, dtype=int)
    bins[order] = (np.arange(n) * n_bins) // n
    return bins


def compute_optimal(bins, outcomes, n_bins=5):
    """Return (best strategy name per bin, overall accuracy of per-bin choices)."""
    bins = np.asarray(bins)
    choices = []
    n_correct = 0.0
    for b in range(n_bins):
        mask = bins == b
        best_name, best_mean = None, -1.0
        for name, out in outcomes.items():
            mean = float(np.mean(np.asarray(out)[mask]))
            if mean > best_mean:
                best_name, best_mean = name, mean
        choices.append(best_name)
        n_correct += float(np.sum(np.asarray(outcomes[best_name])[mask]))
    return choices, n_correct / len(bins)
