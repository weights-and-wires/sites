# Three ways to set the incentive, same toy. Compare the learned team size.
import numpy as np

P, SETUP, MAX_N = 0.9, 0.6, 8


def speed(n):
    return 1.0 / ((1 - P) + P / n)


def progress(n, tmax):
    return speed(n) * max(0.0, tmax - SETUP * (n - 1))


def finish_time(n, work=100.0):
    return SETUP * (n - 1) + work / speed(n)


def softmax(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()


def train(reward, steps=300, group=16, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros(MAX_N)
    for step in range(steps):
        pr = softmax(theta)
        ns = rng.choice(MAX_N, size=group, p=pr) + 1
        r = np.array([reward(n, step) for n in ns])
        adv = (r - r.mean()) / (r.std() + 1e-8)
        g = np.zeros(MAX_N)
        for n, a in zip(ns, adv):
            one = np.zeros(MAX_N)
            one[n - 1] = 1.0
            g += a * (one - pr)
        theta += lr * g / group
    return softmax(theta)


def mean_n(pr):
    return float(pr @ np.arange(1, MAX_N + 1))


# 1. Task reward only: every team size completes the task, so every reward is equal.
print("completion only          E[n] =", round(mean_n(train(lambda n, s: 1.0)), 2), "(uniform is 4.50)")
# 2. An explicit latency penalty: 1 - beta * T / Tmax, with beta 0.5 and Tmax 100.
print("latency penalty          E[n] =", round(mean_n(train(lambda n, s: 1.0 - 0.5 * finish_time(n) / 100.0)), 2))
# 3. A deadline of 5, then 20, then a ramp from 2 to 20 over training.
print("deadline 5               E[n] =", round(mean_n(train(lambda n, s: progress(n, 5))), 2))
print("deadline 20              E[n] =", round(mean_n(train(lambda n, s: progress(n, 20))), 2))
print("deadline ramped 2 -> 20  E[n] =", round(mean_n(train(lambda n, s: progress(n, 2 + 18 * s / 299))), 2))
# 4. The ramp across six seeds: the policy can lock in a small team early. Once all
# 16 samples in a group use the same team size, every advantage is 0 and learning stops.
ramp = [round(mean_n(train(lambda n, s: progress(n, 2 + 18 * s / 299), seed=sd)), 2) for sd in range(6)]
print("ramp, seeds 0 to 5       E[n] =", ramp)
