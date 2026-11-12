import numpy as np


def reinforce_gradient(states, actions, rewards, theta, gamma):
    """Return the mean over episodes of sum_t gamma^t G_t grad log pi(a_t | s_t), shape (S, A).

    states, actions, rewards: arrays of shape (N, H).
    theta: logits of a tabular softmax policy, shape (S, A).
    G_t is the discounted reward-to-go from step t within the episode.
    """
    n, h = states.shape
    z = theta - theta.max(axis=1, keepdims=True)
    pi = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    ret = np.zeros((n, h))
    acc = np.zeros(n)
    for t in range(h - 1, -1, -1):
        acc = rewards[:, t] + gamma * acc
        ret[:, t] = acc
    grad = np.zeros_like(theta, dtype=float)
    for t in range(h):
        s, a = states[:, t], actions[:, t]
        coef = gamma**t * ret[:, t]
        np.add.at(grad, (s, a), coef)
        for b in range(theta.shape[1]):
            np.add.at(grad[:, b], s, -coef * pi[s, b])
    return grad / n
