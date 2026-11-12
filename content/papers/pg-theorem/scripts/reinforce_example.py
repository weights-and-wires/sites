#!/usr/bin/env python3
"""REINFORCE (score function) estimate versus the exact gradient (Section 1, after Equation 2)."""

import numpy as np
from pg_lib import policy_gradient, random_mdp, softmax_rows

np.set_printoptions(precision=4, suppress=True)


def sample(P, theta, s0, n, horizon, rng):
    pi = softmax_rows(theta)
    S, A = theta.shape
    states = np.zeros((n, horizon), dtype=int)
    actions = np.zeros((n, horizon), dtype=int)
    s = np.full(n, s0)
    cp = np.cumsum(pi, axis=1)
    cP = np.cumsum(P, axis=2)
    for t in range(horizon):
        states[:, t] = s
        a = (rng.random(n)[:, None] > cp[s]).sum(axis=1)
        actions[:, t] = a
        s = (rng.random(n)[:, None] > cP[s, a]).sum(axis=1)
    return states, actions


def estimate(R, theta, gamma, states, actions, use_gamma_t=True):
    n, h = states.shape
    pi = softmax_rows(theta)
    r = R[states, actions]
    ret = np.zeros_like(r)
    acc = np.zeros(n)
    for t in range(h - 1, -1, -1):
        acc = r[:, t] + gamma * acc
        ret[:, t] = acc
    disc = gamma ** np.arange(h) if use_gamma_t else np.ones(h)
    grad = np.zeros_like(theta)
    for t in range(h):
        s, a = states[:, t], actions[:, t]
        coef = disc[t] * ret[:, t]
        np.add.at(grad, (s, a), coef)
        for b in range(grad.shape[1]):
            np.add.at(grad[:, b], s, -coef * pi[s, b])
    return grad / n


if __name__ == "__main__":
    P, R = random_mdp(3, 2, seed=0)
    theta = np.random.default_rng(1).normal(size=(3, 2))
    gamma, s0, horizon = 0.9, 0, 80
    exact = policy_gradient(P, R, theta, gamma, s0)
    print("exact gradient:\n", exact)
    rng = np.random.default_rng(0)
    for n in (100, 1000, 20000):
        st, ac = sample(P, theta, s0, n, horizon, rng)
        est = estimate(R, theta, gamma, st, ac)
        print(f"n = {n:5d}: max abs error {np.abs(est - exact).max():.4f}")
    st, ac = sample(P, theta, s0, 20000, horizon, np.random.default_rng(0))
    no_g = estimate(R, theta, gamma, st, ac, use_gamma_t=False)
    print(f"without the gamma^t factor, n = 20000: max abs error {np.abs(no_g - exact).max():.4f}")
