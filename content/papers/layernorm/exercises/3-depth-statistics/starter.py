import numpy as np


def layer_norm(a, eps=1e-5):
    """Finished for you: gain 1, bias 0, statistics over the last axis."""
    mu = a.mean(axis=-1, keepdims=True)
    var = a.var(axis=-1, keepdims=True)
    return (a - mu) / np.sqrt(var + eps)


def layer_stds(x, weights, use_ln):
    """Return an array of shape (L, N): the per-example std of the summed inputs at each layer.

    x:       (N, H) inputs
    weights: list of L matrices, each of shape (H, H)
    use_ln:  normalize the summed inputs before measuring and before the ReLU
    """
    raise NotImplementedError
