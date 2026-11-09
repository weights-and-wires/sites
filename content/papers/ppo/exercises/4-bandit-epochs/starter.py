import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def clip_grad_logp(logp_new, logp_old, adv, eps):
    """Finished for you: gradient of mean(L_CLIP) with respect to logp_new.

    eps=None gives the gradient of mean(r * A), the unclipped L_CPI.
    """
    ratio = np.exp(logp_new - logp_old)
    n = len(ratio)
    if eps is None:
        return ratio * adv / n
    active = ratio * adv <= np.clip(ratio, 1 - eps, 1 + eps) * adv
    return np.where(active, ratio * adv, 0.0) / n


def ppo_epochs(theta_old, actions, adv, eps, lr, epochs):
    """Return the logits after `epochs` full-batch ascent steps on L_CLIP.

    theta_old: float array (K,). actions: int array (N,). adv: float array (N,).
    """
    raise NotImplementedError
