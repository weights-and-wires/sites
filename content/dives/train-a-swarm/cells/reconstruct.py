# Learn a team size from a deadline reward. Our toy, not the post's training run.
import numpy as np

P, SETUP = 0.9, 0.6
MAX_N = 8


def progress(n, tmax):
    speed = 1.0 / ((1 - P) + P / n)
    return speed * max(0.0, tmax - SETUP * (n - 1))


def softmax(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()


def run(tmax, steps=300, group=16, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros(MAX_N)
    for _ in range(steps):
        pr = softmax(theta)
        ns = rng.choice(MAX_N, size=group, p=pr) + 1
        r = np.array([progress(n, tmax) for n in ns])
        adv = (r - r.mean()) / (r.std() + 1e-8)  # one flat advantage per rollout
        g = np.zeros(MAX_N)
        for n, a in zip(ns, adv):
            one = np.zeros(MAX_N)
            one[n - 1] = 1.0
            g += a * (one - pr)
        theta += lr * g / group
    return softmax(theta)


tmaxes = [2, 5, 10, 20, 40]
probs = []
for tmax in tmaxes:
    pr = run(tmax)
    best = max(range(1, MAX_N + 1), key=lambda n: progress(n, tmax))
    probs.append([round(float(x), 4) for x in pr])
    print(f"Tmax={tmax:>3}  best n={best}  learned E[n]={pr @ np.arange(1, MAX_N + 1):.2f}  mode={pr.argmax() + 1}")

result = {"tmax": tmaxes, "probs": probs}
