import numpy as np


def reinforce_gradient(states, actions, rewards, theta, gamma):
    """Return the mean over episodes of sum_t gamma^t G_t grad log pi(a_t | s_t), shape (S, A).

    states, actions, rewards: arrays of shape (N, H).
    theta: logits of a tabular softmax policy, shape (S, A).
    G_t is the discounted reward-to-go from step t within the episode.
    """
    raise NotImplementedError
