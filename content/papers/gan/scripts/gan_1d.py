"""Train a 1-D GAN with manual gradients: G(z) = mu + exp(s) z, D(x) = sigmoid(w1 x + w2 x^2 + b).

The data are N(3, 1). The script uses the non-saturating generator loss from
Section 3 and compares k = 5 discriminator steps per generator step (lr 0.02,
4000 iterations) against k = 1 (lr 0.05, 2000 iterations).

Run: python3 scripts/gan_1d.py
"""

import numpy as np


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def feats(x):
    return np.stack([x, x**2, np.ones_like(x)], axis=1)


def d_grad(pd, x_real, x_fake):
    m = len(x_real)
    g_real = feats(x_real).T @ (-(1 - sigmoid(feats(x_real) @ pd))) / m
    g_fake = feats(x_fake).T @ sigmoid(feats(x_fake) @ pd) / m
    return g_real + g_fake


def g_grad(pg, pd, z):
    mu, s = pg
    x = mu + np.exp(s) * z
    a = feats(x) @ pd
    dx = -(1 - sigmoid(a)) * (pd[0] + 2 * pd[1] * x) / len(z)
    return np.array([dx.sum(), (dx * np.exp(s) * z).sum()])


def train(seed, steps, lr, k, m=64):
    rng = np.random.default_rng(seed)
    pd, pg = np.zeros(3), np.zeros(2)
    for _ in range(steps):
        for _ in range(k):
            x_real = 3.0 + rng.normal(size=m)
            x_fake = pg[0] + np.exp(pg[1]) * rng.normal(size=m)
            pd = pd - lr * d_grad(pd, x_real, x_fake)
        pg = pg - lr * g_grad(pg, pd, rng.normal(size=m))
    return pg


for name, steps, lr, k in [("k=5, lr=0.02", 4000, 0.02, 5), ("k=1, lr=0.05", 2000, 0.05, 1)]:
    print(name)
    for seed in range(3):
        mu, s = train(seed, steps, lr, k)
        print(f"  seed {seed}: mu = {mu:.2f}, sigma = {np.exp(s):.2f}")
print("target: mu = 3.00, sigma = 1.00")
