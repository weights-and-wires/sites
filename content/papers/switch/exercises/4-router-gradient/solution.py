import numpy as np


def softmax(z):
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def switch_forward(x, Wr, W):
    """x: (T, d), Wr: (d, N), W: (N, d, d_out).

    Returns (y, cache) where y[t] = p[t, i_t] * (x[t] @ W[i_t]) and i_t is
    the argmax expert of token t. cache holds what switch_backward needs.
    """
    T = len(x)
    p = softmax(x @ Wr)
    idx = np.argmax(p, axis=-1)
    gate = p[np.arange(T), idx]
    out = np.einsum("td,tdo->to", x, W[idx])
    return gate[:, None] * out, (x, p, idx, gate, out, W.shape)


def switch_backward(cache, dy):
    """dy: (T, d_out), the gradient of the loss with respect to y.

    Returns (dWr, dW), with the shapes of Wr and W.
    """
    x, p, idx, gate, out, W_shape = cache
    T = len(x)
    dW = np.zeros(W_shape)
    d_out_act = dy * gate[:, None]
    for i in range(W_shape[0]):
        m = idx == i
        dW[i] = x[m].T @ d_out_act[m]
    dgate = np.sum(dy * out, axis=1)
    dp = np.zeros_like(p)
    dp[np.arange(T), idx] = dgate
    dh = p * (dp - np.sum(dp * p, axis=1, keepdims=True))
    return x.T @ dh, dW


def train(x, y_target, N, steps=2000, lr=0.05, seed=0):
    """Finished for you: full-batch gradient descent on the mean squared error."""
    rng = np.random.default_rng(seed)
    d, d_out = x.shape[1], y_target.shape[1]
    Wr = 0.1 * rng.normal(size=(d, N))
    W = 0.1 * rng.normal(size=(N, d, d_out))
    for _ in range(steps):
        y, cache = switch_forward(x, Wr, W)
        dy = 2 * (y - y_target) / y.size * d_out
        dWr, dW = switch_backward(cache, dy)
        Wr -= lr * dWr
        W -= lr * dW
    y, cache = switch_forward(x, Wr, W)
    return Wr, W, float(np.mean((y - y_target) ** 2))
