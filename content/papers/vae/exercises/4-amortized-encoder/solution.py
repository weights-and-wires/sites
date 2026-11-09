import numpy as np


def neg_elbo(w, c, x, eps):
    """Finished for you: mean over the batch of -ELBO (constants included).

    Encoder: mu = w * x, log sigma^2 = c. z = mu + exp(c / 2) * eps.
    x, eps: arrays of shape (batch,).
    """
    mu = w * x
    z = mu + np.exp(0.5 * c) * eps
    kl = 0.5 * (mu**2 + np.exp(c) - 1 - c)
    rec = 0.5 * np.log(2 * np.pi) + 0.5 * (x - z) ** 2
    return np.mean(kl + rec)


def grad_neg_elbo(w, c, x, eps):
    """Return (dw, dc): the gradient of neg_elbo with respect to w and c, eps fixed."""
    sigma = np.exp(0.5 * c)
    mu = w * x
    z = mu + sigma * eps
    d_mu = mu - (x - z)
    d_c = 0.5 * (np.exp(c) - 1) - (x - z) * eps * 0.5 * sigma
    return np.mean(d_mu * x), np.mean(d_c)


def fit(steps=3000, lr=0.02, batch=128, seed=0):
    """Finished for you: SGD on the negative ELBO with x drawn from p(x) = N(0, 2)."""
    rng = np.random.default_rng(seed)
    w, c = 0.0, 0.0
    for _ in range(steps):
        x = np.sqrt(2.0) * rng.standard_normal(batch)
        eps = rng.standard_normal(batch)
        dw, dc = grad_neg_elbo(w, c, x, eps)
        w -= lr * dw
        c -= lr * dc
    return w, c
