import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def make_net(depth, d, seed, residual, gain=0.5):
    """Finished for you. Returns a list of layers.

    Plain net: `depth` layers, each a single matrix W, with h -> relu(W h).
    Residual net: `depth // 2` blocks, each a pair (W1, W2), with
    h -> relu(W2 relu(W1 h) + h), as in Figure 2 of the paper.
    Both have the same number of weight matrices. Weights are N(0, (gain^2 * 2 / d)).
    """
    rng = np.random.default_rng(seed)
    std = gain * np.sqrt(2.0 / d)
    mats = [rng.normal(0, std, size=(d, d)) for _ in range(depth)]
    if residual:
        return [(mats[2 * i], mats[2 * i + 1]) for i in range(depth // 2)]
    return mats


def layer_forward(layer, h, residual):
    """Finished for you. Returns (output, cache)."""
    if residual:
        W1, W2 = layer
        a = W1 @ h
        z = W2 @ relu(a) + h
        return relu(z), (h, a, z)
    z = layer @ h
    return relu(z), (h, z)


def layer_backward(layer, cache, g_out, residual):
    """Finished for you. Returns the gradient of the loss with respect to the layer input."""
    if residual:
        W1, W2 = layer
        h, a, z = cache
        g_z = g_out * (z > 0)
        g_a = (W2.T @ g_z) * (a > 0)
        return W1.T @ g_a + g_z
    h, z = cache
    return layer.T @ (g_out * (z > 0))


def input_grad_norms(net, x, residual):
    """Return an array of n + 1 norms, where n = len(net).

    Entry l is the Euclidean norm of dL/dh_l for L = 0.5 * ||h_n||^2,
    where h_0 = x and h_l is the output of layer l.
    """
    raise NotImplementedError
