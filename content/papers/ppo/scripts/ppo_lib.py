"""Shared NumPy helpers for the PPO page scripts.

These mirror the solutions of the page's exercises. Notation follows the paper:
r_t(theta) is the probability ratio (Section 3), L^CLIP is Equation 7.
"""

import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def l_cpi(ratio, adv):
    """Per-sample conservative policy iteration term r * A (Equation 6)."""
    return ratio * adv


def l_clip(ratio, adv, eps):
    """Per-sample clipped term, min(r A, clip(r, 1-eps, 1+eps) A) (Equation 7)."""
    return np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)


def clip_grad_logp(logp_new, logp_old, adv, eps):
    """Gradient of mean(L^CLIP) with respect to logp_new. eps=None gives L^CPI."""
    ratio = np.exp(logp_new - logp_old)
    n = len(ratio)
    if eps is None:
        return ratio * adv / n
    active = ratio * adv <= np.clip(ratio, 1 - eps, 1 + eps) * adv
    return np.where(active, ratio * adv, 0.0) / n


def ppo_epochs(theta_old, actions, adv, eps, lr, epochs):
    """K epochs of full-batch gradient ascent on L^CLIP for a softmax bandit policy."""
    k = len(theta_old)
    logp_old = np.log(softmax(theta_old))[actions]
    onehot = np.eye(k)[actions]
    theta = np.array(theta_old, dtype=float)
    for _ in range(epochs):
        pi = softmax(theta)
        g = clip_grad_logp(np.log(pi)[actions], logp_old, adv, eps)
        theta = theta + lr * (g @ (onehot - pi))
    return theta


def kl_categorical(p_old, p_new):
    """Mean over rows of KL[p_old, p_new] (the direction in Equation 8)."""
    p_old, p_new = np.atleast_2d(p_old), np.atleast_2d(p_new)
    return float(np.mean(np.sum(p_old * (np.log(p_old) - np.log(p_new)), axis=-1)))


def update_beta(beta, d, d_targ):
    """The adaptive rule below Equation 8."""
    if d < d_targ / 1.5:
        return beta / 2
    if d > d_targ * 1.5:
        return beta * 2
    return beta
