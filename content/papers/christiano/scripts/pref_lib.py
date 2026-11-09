"""Shared NumPy helpers for the Christiano page scripts.

These mirror the solutions of the page's exercises. Notation follows the paper
(Section 2.2.3): r_hat is the learned reward, a segment sigma is a sequence of
(observation, action) pairs, and P_hat[sigma1 > sigma2] is Equation 1. The
paper calls the human label mu; here mu is the probability mass on sigma1.
"""

import numpy as np


def log_sigmoid(x):
    """log(1 / (1 + exp(-x))) without overflow."""
    return -np.logaddexp(0.0, -np.asarray(x, dtype=float))


def sigmoid(x):
    return np.exp(log_sigmoid(x))


def pref_prob(ret1, ret2, eps=0.0):
    """P_hat[sigma1 > sigma2] (Equation 1) with the 10% error adjustment of Section 2.2.3."""
    p = sigmoid(np.asarray(ret1, dtype=float) - np.asarray(ret2, dtype=float))
    return (1.0 - eps) * p + eps / 2.0


def pref_loss(ret1, ret2, mu, eps=0.0):
    """Mean cross-entropy between mu and P_hat (the loss below Equation 1)."""
    ret1 = np.asarray(ret1, dtype=float)
    ret2 = np.asarray(ret2, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if eps == 0.0:
        lp1 = log_sigmoid(ret1 - ret2)
        lp2 = log_sigmoid(ret2 - ret1)
    else:
        lp1 = np.log(pref_prob(ret1, ret2, eps))
        lp2 = np.log(pref_prob(ret2, ret1, eps))
    return float(np.mean(-(mu * lp1 + (1.0 - mu) * lp2)))


def seg_sum(F, w):
    """Segment returns for a linear reward r_hat(o, a) = w . phi(o, a). F has shape (n, k, d)."""
    return F.sum(axis=1) @ w


def loss_grad(w, F1, F2, mu):
    d = seg_sum(F1, w) - seg_sum(F2, w)
    return ((sigmoid(d) - mu)[:, None] * (F1.sum(axis=1) - F2.sum(axis=1))).mean(axis=0)


def fit_reward(F1, F2, mu, lr=0.5, steps=400, l2=0.0):
    w = np.zeros(F1.shape[-1])
    for _ in range(steps):
        w = w - lr * (loss_grad(w, F1, F2, mu) + l2 * w)
    return w


def normalize_rewards(r):
    r = np.asarray(r, dtype=float)
    sd = r.std()
    return (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)


def ensemble_reward(ws, F):
    return np.mean([normalize_rewards(F @ w) for w in ws], axis=0)


def select_queries(ws, F1, F2, m):
    probs = np.array([sigmoid(seg_sum(F1, w) - seg_sum(F2, w)) for w in ws])
    var = probs.var(axis=0)
    return np.argsort(-var, kind="stable")[:m]


def make_pairs(rng, n, k, d, w_true, noise=0.0):
    """Synthetic segments with a linear true reward; labels from the true returns."""
    F1 = rng.normal(size=(n, k, d)) + rng.normal(size=(n, 1, d))
    F2 = rng.normal(size=(n, k, d)) + rng.normal(size=(n, 1, d))
    prefer1 = (seg_sum(F1, w_true) > seg_sum(F2, w_true)).astype(float)
    if noise > 0:
        flip = rng.random(n) < noise
        prefer1 = np.where(flip, rng.integers(0, 2, n).astype(float), prefer1)
    return F1, F2, prefer1
