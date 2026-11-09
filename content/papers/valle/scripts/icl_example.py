"""Toy in-context speaker inference: a Bayesian mixture over speakers (Section 4.3)."""
import numpy as np

V, K = 6, 3
trans = np.full((K, V, V), 0.3 / (V - 1))
for k in range(K):
    for v in range(V):
        trans[k, v, (v + k + 1) % V] = 0.7  # speaker k steps by k + 1 mod V
prior = np.ones(K) / K


def posterior(prompt):
    logp = np.log(prior).copy()
    for a, b in zip(prompt[:-1], prompt[1:]):
        logp += np.log(trans[:, a, b])
    p = np.exp(logp - logp.max())
    return p / p.sum()


def next_dist(prefix):
    return posterior(prefix) @ trans[:, prefix[-1], :]


speaker1 = [(2 * t) % V for t in range(12)]
for n in [1, 2, 3, 5, 9]:
    print(f"prompt length {n}: posterior {np.round(posterior(speaker1[:n]), 4).tolist()}")
print("next-token distribution after [0, 2]:", np.round(next_dist([0, 2]), 4).tolist())
print("next-token distribution after [0]:", np.round(next_dist([0]), 4).tolist())
seq = [0, 2]
for _ in range(5):
    seq.append(int(np.argmax(next_dist(seq))))
print("greedy continuation of [0, 2]:", seq)
long = [(2 * t) % V for t in range(2000)]
print("posterior after 2000 tokens:", np.round(posterior(long), 4).tolist())
