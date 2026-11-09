import numpy as np


def clip_term(ratio, adv, eps):
    """Finished for you: per-sample min(r*A, clip(r, 1-eps, 1+eps)*A)."""
    return np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)


def entropy(probs):
    """Entropy in nats of each row of probs (shape (N, K)); returns shape (N,)."""
    probs = np.asarray(probs, dtype=float)
    return -np.sum(probs * np.log(probs), axis=-1)


def ppo_objective(probs, probs_old, actions, adv, values, returns, eps, c1, c2):
    """Equation 9 as a scalar: mean of L_CLIP - c1 * L_VF + c2 * S."""
    idx = np.arange(len(actions))
    ratio = probs[idx, actions] / probs_old[idx, actions]
    l_clip = clip_term(ratio, adv, eps)
    l_vf = (values - returns) ** 2
    return float(np.mean(l_clip - c1 * l_vf + c2 * entropy(probs)))


def ppo_loss(probs, probs_old, actions, adv, values, returns, eps, c1, c2):
    """The scalar to minimize: the negative of ppo_objective."""
    return -ppo_objective(probs, probs_old, actions, adv, values, returns, eps, c1, c2)
