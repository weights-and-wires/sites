import numpy as np

EPS = 0.001


def h(x):
    """Return sign(x) * (sqrt(|x| + 1) - 1) + EPS * x."""
    raise NotImplementedError


def h_inv(y):
    """Return the exact inverse of h, so that h_inv(h(x)) == x."""
    raise NotImplementedError


def phi(x, lo=-300, hi=300):
    """Return a vector of length hi - lo + 1 with the weights of the two integers around x."""
    raise NotImplementedError


def to_target(x, lo=-300, hi=300):
    """Return phi(h(x)): the categorical training target for a scalar x."""
    raise NotImplementedError


def from_logits(logits, lo=-300):
    """Softmax the logits, take the expected support value, then invert h. Return a float."""
    raise NotImplementedError
