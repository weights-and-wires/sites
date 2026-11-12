"""Shared code for the train-a-swarm toy. This model is ours, not the post's."""

import numpy as np


def speed(n, p=0.9, rate=1.0):
    """Amdahl's law: work units per time unit for a team of n agents."""
    return rate / ((1 - p) + p / n)


def progress(n, tmax, p=0.9, setup=0.6):
    """Work done by the deadline: the team pays a setup tax before it works."""
    return speed(n, p) * max(0.0, tmax - setup * (n - 1))


def finish_time(n, work=100.0, p=0.9, setup=0.6):
    """Time to finish `work` units with no deadline."""
    return setup * (n - 1) + work / speed(n, p)


def softmax(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()


def train(reward, steps=300, group=16, lr=0.5, seed=0, max_n=8):
    """REINFORCE with a flat, group-relative advantage over team size.

    reward(n, step) returns the scalar reward for a rollout with n agents.
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros(max_n)
    for step in range(steps):
        pr = softmax(theta)
        ns = rng.choice(max_n, size=group, p=pr) + 1
        r = np.array([reward(n, step) for n in ns])
        adv = (r - r.mean()) / (r.std() + 1e-8)
        g = np.zeros(max_n)
        for n, a in zip(ns, adv):
            one = np.zeros(max_n)
            one[n - 1] = 1.0
            g += a * (one - pr)
        theta += lr * g / group
    return softmax(theta)


def mean_n(pr):
    return float(pr @ np.arange(1, len(pr) + 1))
