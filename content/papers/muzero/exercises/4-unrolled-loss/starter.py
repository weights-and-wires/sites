import numpy as np


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def unroll(params, obs, actions):
    """Return (rewards, policies, values) from unrolling h, g, f along `actions`.

    params holds the matrices H, G, Wp and the vector wv. rewards has K entries
    (steps 1..K); policies and values have K + 1 entries (steps 0..K).
    """
    raise NotImplementedError


def unrolled_loss(params, obs, actions, pi, z, u):
    """Return the sum over k = 0..K of reward, value, and policy losses (no L2 term)."""
    raise NotImplementedError
