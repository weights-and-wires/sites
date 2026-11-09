#!/usr/bin/env python3
"""Why the reward model and the policy train together (Section 3.3, "no online queries").

A toy with K arms on a line, x in [0, 1]. The true reward peaks at x = 0.5 and falls
on both sides. The reward model is linear in x, so it can only say "more is better"
or "less is better". Only the simulated labeler sees the true reward.

Offline: all queries come from the initial policy, which sits on the rising side.
Online: queries come from the current policy, in rounds, while it improves.
"""

import numpy as np
from pref_lib import sigmoid

K = 21
x = np.linspace(0.0, 1.0, K)
true_r = 1.0 - 4.0 * (x - 0.5) ** 2  # 0 at the ends, 1 at the peak
BUDGET = 400
ROUNDS = 8


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def fit_slope(pairs, labels, a0, steps=300, lr=2.0):
    """Linear reward r_hat(x) = a * x fitted with the Bradley-Terry loss."""
    a = a0
    for _ in range(steps):
        d = a * (x[pairs[:, 0]] - x[pairs[:, 1]])
        g = np.mean((sigmoid(d) - labels) * (x[pairs[:, 0]] - x[pairs[:, 1]]))
        a -= lr * g
    return a


def sample_pairs(rng, policy, n):
    a = rng.choice(K, size=n, p=policy)
    b = rng.choice(K, size=n, p=policy)
    keep = a != b
    return np.stack([a[keep], b[keep]], axis=1)


def label(pairs):
    return (true_r[pairs[:, 0]] > true_r[pairs[:, 1]]).astype(float)


def improve_policy(logits, r_hat, steps, lr):
    """Natural-gradient ascent on E_pi[r_hat] for a softmax policy, a stand-in for the RL step."""
    z = logits.copy()
    for _ in range(steps):
        p = softmax(z)
        z = z + lr * (r_hat - p @ r_hat)
    return z


def expected_true(logits):
    return float(softmax(logits) @ true_r)


def initial_logits():
    return -((x - 0.2) ** 2) / (2 * 0.08**2)


def run(online, seed):
    rng = np.random.default_rng(seed)
    logits = initial_logits()
    a = 0.0
    pairs = np.zeros((0, 2), dtype=int)
    labels = np.zeros(0)
    per_round = BUDGET // ROUNDS
    if not online:
        pairs = sample_pairs(rng, softmax(logits), BUDGET)
        labels = label(pairs)
        a = fit_slope(pairs, labels, a)
    slopes = []
    for _ in range(ROUNDS):
        if online:
            new = sample_pairs(rng, softmax(logits), per_round)
            pairs = np.concatenate([pairs, new])
            labels = np.concatenate([labels, label(new)])
            a = fit_slope(pairs, labels, a)
        slopes.append(a)
        logits = improve_policy(logits, a * x, steps=40, lr=0.05)
    return logits, slopes


start = initial_logits()
print(f"initial policy: mean x {softmax(start) @ x:.3f}, expected true reward {expected_true(start):+.3f}")
print(f"{BUDGET} queries in total, {ROUNDS} policy rounds, mean over 20 seeds")
for online in [False, True]:
    vals, means, last = [], [], []
    for seed in range(20):
        logits, slopes = run(online, seed)
        vals.append(expected_true(logits))
        means.append(softmax(logits) @ x)
        last.append(slopes[-1])
    name = "online " if online else "offline"
    print(f"{name}: final mean x {np.mean(means):.3f}, final slope {np.mean(last):+.2f}, "
          f"expected true reward {np.mean(vals):+.3f}")
