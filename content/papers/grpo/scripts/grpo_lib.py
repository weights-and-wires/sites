"""Shared NumPy helpers for the GRPO page scripts.

These mirror the solutions of the page's exercises. Notation follows the paper
(DeepSeekMath, Section 4.1): G is the group size, A_hat is the advantage,
epsilon is the clip range, beta is the KL coefficient.
"""

import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def group_advantages(rewards, eps=1e-6):
    """Outcome-supervision advantage (Section 4.1.2): (r - mean) / std per group.

    The last axis is the group. std is the population standard deviation, and
    eps keeps a group of identical rewards at advantage 0 instead of 0/0.
    """
    r = np.asarray(rewards, dtype=float)
    mean = r.mean(axis=-1, keepdims=True)
    std = r.std(axis=-1, keepdims=True)
    return (r - mean) / (std + eps)


def kl_k3(logp, logp_ref):
    """The estimator of Equation 4: pi_ref/pi - log(pi_ref/pi) - 1."""
    d = logp_ref - logp
    return np.exp(d) - d - 1.0


def token_terms(logp_new, logp_old, adv, eps):
    """min(r A, clip(r, 1-eps, 1+eps) A) per token, the clipped part of Equation 3."""
    ratio = np.exp(logp_new - logp_old)
    a = adv[:, None]
    return np.minimum(ratio * a, np.clip(ratio, 1 - eps, 1 + eps) * a)


def grpo_objective(logp_new, logp_old, logp_ref, adv, mask, eps, beta):
    """Equation 3 for one group, without the outer expectation over questions.

    All arrays except adv have shape (G, T); adv has shape (G,); mask is 1 for
    real tokens and 0 for padding.
    """
    mask = np.asarray(mask, dtype=float)
    per_token = token_terms(logp_new, logp_old, adv, eps) - beta * kl_k3(logp_new, logp_ref)
    per_output = (per_token * mask).sum(axis=1) / mask.sum(axis=1)
    return per_output.mean()


def process_advantages(step_rewards):
    """Process-supervision advantages (Section 4.1.3), one value per step.

    step_rewards is a list of G 1-D arrays, one reward per reasoning step. The
    rewards are normalized with the mean and std of all steps of all outputs,
    and the advantage of step j is the sum of the normalized rewards of steps
    j, j+1, ..., K_i.
    """
    flat = np.concatenate([np.asarray(s, dtype=float) for s in step_rewards])
    mean, std = flat.mean(), flat.std()
    out = []
    for s in step_rewards:
        norm = (np.asarray(s, dtype=float) - mean) / (std + 1e-6)
        out.append(np.cumsum(norm[::-1])[::-1])
    return out


def grpo_update(logits, actions, rewards, logits_ref, lr, beta):
    """One gradient-ascent step on Equation 3 for a one-token softmax policy.

    With a single update per batch the ratio is 1, so the gradient is Equation 20:
    mean_i (A_i + beta (pi_ref(a_i)/pi(a_i) - 1)) * (onehot(a_i) - pi).
    """
    pi = softmax(logits)
    pi_ref = softmax(logits_ref)
    adv = group_advantages(rewards)
    coef = adv + beta * (pi_ref[actions] / pi[actions] - 1.0)
    onehot = np.eye(len(logits))[actions]
    grad = (coef[:, None] * (onehot - pi)).mean(axis=0)
    return logits + lr * grad
