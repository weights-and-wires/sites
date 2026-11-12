import numpy as np


def best_of_n_accuracy(n, p, sep, sigma, trials, rng):
    """Accuracy of best-of-n selection with a noisy scorer.

    In each trial, draw n solutions. Each is correct with probability p. The scorer
    gives a solution the score sep * correct + sigma * noise, with noise drawn from
    N(0, 1). Keep the solution with the highest score and count the trial as a hit if
    that solution is correct. Return the fraction of hits as a float.

    `rng` is a numpy.random.Generator.
    """
    raise NotImplementedError


def majority_vote_accuracy(n, p, n_wrong, trials, rng):
    """Accuracy of plurality voting over n solutions.

    Each solution is correct with probability p. A wrong solution carries one of
    n_wrong wrong answers, each equally likely. Return the fraction of trials in which
    the correct answer has the most votes. Break ties in favor of the answer that
    appears first. Return a float.
    """
    raise NotImplementedError
