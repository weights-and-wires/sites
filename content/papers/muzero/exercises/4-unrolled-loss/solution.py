import numpy as np


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def unroll(params, obs, actions):
    """Return (rewards, policies, values) from unrolling h, g, f along `actions`.

    rewards has K entries (steps 1..K); policies and values have K + 1 entries.
    """
    H, G, Wp, wv = params["H"], params["G"], params["Wp"], params["wv"]
    A = Wp.shape[0]
    s = H @ obs
    rewards, policies, values = [], [softmax(Wp @ s)], [float(wv @ s)]
    for a in actions:
        out = G @ np.concatenate([s, np.eye(A)[a]])
        rewards.append(float(out[0]))
        s = out[1:]
        policies.append(softmax(Wp @ s))
        values.append(float(wv @ s))
    return rewards, policies, values


def unrolled_loss(params, obs, actions, pi, z, u):
    """Return the sum over k = 0..K of reward, value, and policy losses."""
    rewards, policies, values = unroll(params, obs, actions)
    total = 0.0
    for k in range(len(actions) + 1):
        if k > 0:
            total += (u[k] - rewards[k - 1]) ** 2
        total += (z[k] - values[k]) ** 2
        total += -float(np.asarray(pi[k]) @ np.log(policies[k]))
    return float(total)
