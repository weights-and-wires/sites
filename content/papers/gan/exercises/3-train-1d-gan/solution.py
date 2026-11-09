import numpy as np


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def features(x):
    """Return an (m, 3) matrix with columns [x, x^2, 1]."""
    return np.stack([x, x**2, np.ones_like(x)], axis=1)


def d_grad(pd, x_real, x_fake):
    """Gradient of -mean[log D(x_real) + log(1 - D(x_fake))] with respect to pd = [w1, w2, b]."""
    m = len(x_real)
    d_real = sigmoid(features(x_real) @ pd)
    d_fake = sigmoid(features(x_fake) @ pd)
    return features(x_real).T @ (-(1 - d_real)) / m + features(x_fake).T @ d_fake / m


def g_grad(pg, pd, z):
    """Gradient of -mean[log D(G(z))] with respect to pg = [mu, s], where G(z) = mu + exp(s) * z."""
    mu, s = pg
    x = mu + np.exp(s) * z
    d = sigmoid(features(x) @ pd)
    dx = -(1 - d) * (pd[0] + 2 * pd[1] * x) / len(z)
    return np.array([dx.sum(), (dx * np.exp(s) * z).sum()])


def train(seed=0, steps=4000, lr=0.02, k=5, m=64, data_mean=3.0, data_std=1.0):
    """Algorithm 1 with k discriminator steps per generator step. Returns pg = [mu, s]."""
    rng = np.random.default_rng(seed)
    pd, pg = np.zeros(3), np.zeros(2)
    for _ in range(steps):
        for _ in range(k):
            x_real = data_mean + data_std * rng.normal(size=m)
            x_fake = pg[0] + np.exp(pg[1]) * rng.normal(size=m)
            pd = pd - lr * d_grad(pd, x_real, x_fake)
        pg = pg - lr * g_grad(pg, pd, rng.normal(size=m))
    return pg
