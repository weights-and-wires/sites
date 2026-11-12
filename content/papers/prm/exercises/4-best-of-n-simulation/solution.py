from collections import Counter

import numpy as np


def best_of_n_accuracy(n, p, sep, sigma, trials, rng):
    """Accuracy of best-of-n selection with a scorer of the form sep * correct + sigma * noise."""
    correct = rng.random((trials, n)) < p
    scores = sep * correct + sigma * rng.standard_normal((trials, n))
    pick = scores.argmax(axis=1)
    return float(correct[np.arange(trials), pick].mean())


def majority_vote_accuracy(n, p, n_wrong, trials, rng):
    """Accuracy of plurality voting when wrong answers are spread over n_wrong values."""
    correct = rng.random((trials, n)) < p
    wrong = 1 + rng.integers(0, n_wrong, (trials, n))
    answers = np.where(correct, 0, wrong)
    hits = 0
    for row in answers.tolist():
        hits += Counter(row).most_common(1)[0][0] == 0
    return hits / trials
