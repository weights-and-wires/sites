import numpy as np

NAMES = ("add", "sub", "mul")


def apply_op(value, op):
    """Apply one operation ("add", n), ("sub", n), or ("mul", n) to an integer."""
    name, n = op
    if name == "add":
        return value + n
    if name == "sub":
        return value - n
    return value * n


def make_problem(rng, k):
    """A random start value and k operations. Every operation changes the value."""
    start = int(rng.integers(1, 10))
    ops = []
    for _ in range(k):
        name = NAMES[int(rng.integers(0, 3))]
        n = int(rng.integers(2, 5)) if name == "mul" else int(rng.integers(1, 10))
        ops.append((name, n))
    return start, ops


def answer_only(start, ops, budget, eps=0.0, rng=None):
    """One forward pass: apply only the first `budget` operations, then stop.

    When rng is not None, draw rng.random() once; if it is below eps, add 1 to the result.
    """
    raise NotImplementedError


def stepwise(start, ops, budget, eps=0.0, rng=None):
    """One pass per line: each pass applies up to `budget` operations to the previous line's value.

    A pass that is not the last still writes its value, so an error carries into the next pass.
    Draw rng.random() once per pass when rng is not None; below eps, add 1 to that pass's value.
    Returns (values, answer): the value written after each pass, and the last one (start if no ops).
    """
    raise NotImplementedError


def accuracy(ks, budget, eps, n, seed=0):
    """Return {k: (answer_only_accuracy, stepwise_accuracy)} over n random problems per k.

    Use one np.random.default_rng(seed) for everything. A prediction is right when it equals
    the value of applying all k operations exactly.
    """
    raise NotImplementedError
