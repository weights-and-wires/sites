import numpy as np


def gelu(x):
    """The tanh approximation of GeLU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def all_reduce(parts):
    """Finished for you: sum a list of per-GPU arrays; every GPU receives the sum."""
    return sum(parts)


def split_columns(A, n):
    """Return a list of n equal column blocks of A."""
    raise NotImplementedError


def split_rows(B, n):
    """Return a list of n equal row blocks of B."""
    raise NotImplementedError


def tp_mlp(X, A, B, n, reduce_fn=all_reduce):
    """Tensor-parallel Z = gelu(X A) B on n simulated GPUs, with one all-reduce."""
    raise NotImplementedError
