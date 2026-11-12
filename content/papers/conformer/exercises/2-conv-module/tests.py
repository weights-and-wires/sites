import math
import numpy as np
from submission import conv_module


def sig(v):
    return 1.0 / (1.0 + math.exp(-v))


def reference(x, w1, wd, gamma, beta, w2, eps=1e-5):
    T, d = x.shape
    k = wd.shape[0]
    left = (k - 1) // 2
    xn = np.zeros((T, d))
    for t in range(T):
        m = sum(x[t]) / d
        v = sum((x[t, c] - m) ** 2 for c in range(d)) / d
        for c in range(d):
            xn[t, c] = (x[t, c] - m) / math.sqrt(v + eps)
    g = np.zeros((T, d))
    for t in range(T):
        for c in range(d):
            a = sum(xn[t, i] * w1[i, c] for i in range(d))
            b = sum(xn[t, i] * w1[i, d + c] for i in range(d))
            g[t, c] = a * sig(b)
    z = np.zeros((T, d))
    for t in range(T):
        for c in range(d):
            for i in range(k):
                j = t + i - left
                if 0 <= j < T:
                    z[t, c] += wd[i, c] * g[j, c]
    for c in range(d):
        m = sum(z[:, c]) / T
        v = sum((z[t, c] - m) ** 2 for t in range(T)) / T
        for t in range(T):
            u = (z[t, c] - m) / math.sqrt(v + eps) * gamma[c] + beta[c]
            z[t, c] = u * sig(u)
    return z @ w2


def make(seed, T=7, d=3, k=4, wscale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, d)) * 3 + 1
    w1 = rng.normal(size=(d, 2 * d))
    wd = rng.normal(size=(k, d)) * wscale
    gamma = rng.normal(size=d) + 1
    beta = rng.normal(size=d)
    w2 = rng.normal(size=(d, d))
    return x, w1, wd, gamma, beta, w2


def test_matches_reference():
    args = make(0)
    got = conv_module(*args)
    want = reference(*args)
    assert got.shape == (7, 3), f"expected shape (7, 3), got {got.shape}"
    assert np.allclose(got, want, atol=1e-8), f"expected {want}, got {got}"


def test_matches_reference_odd_kernel():
    args = make(1, T=6, d=4, k=5)
    got = conv_module(*args)
    want = reference(*args)
    assert np.allclose(got, want, atol=1e-8), f"expected {want}, got {got}"


def test_batch_norm_epsilon_matters_for_small_variance():
    args = make(2, wscale=0.01)
    got = conv_module(*args)
    want = reference(*args)
    assert np.allclose(got, want, atol=1e-8), f"expected {want}, got {got}"


def test_batch_norm_statistics_are_over_time():
    # Scaling one channel's depthwise filter cancels in batch norm (eps tiny)
    # only if the statistics are per channel, over time.
    x, w1, wd, gamma, beta, w2 = make(3)
    base = conv_module(x, w1, wd, gamma, beta, w2, eps=1e-12)
    wd2 = wd.copy()
    wd2[:, 1] *= 5.0
    scaled = conv_module(x, w1, wd2, gamma, beta, w2, eps=1e-12)
    assert np.allclose(base, scaled, atol=1e-6), f"expected equal outputs, got max diff {np.abs(base - scaled).max()}"


def test_layer_norm_removes_input_scale():
    x, w1, wd, gamma, beta, w2 = make(4)
    a = conv_module(x, w1, wd, gamma, beta, w2)
    b = conv_module(x * 10.0, w1, wd, gamma, beta, w2)
    assert np.allclose(a, b, atol=1e-3), f"expected the layer norm to cancel a 10x input scale, got max diff {np.abs(a - b).max()}"


def test_glu_gate_is_the_second_half():
    x, w1, wd, gamma, beta, w2 = make(5)
    d = x.shape[1]
    swapped = np.concatenate([w1[:, d:], w1[:, :d]], axis=1)
    a = conv_module(x, w1, wd, gamma, beta, w2)
    b = conv_module(x, swapped, wd, gamma, beta, w2)
    want = reference(x, w1, wd, gamma, beta, w2)
    assert np.allclose(a, want, atol=1e-8), f"expected {want}, got {a}"
    assert not np.allclose(a, b), "expected swapping the two halves of w_pw1 to change the output"
