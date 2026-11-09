#!/usr/bin/env python3
"""How often does a token overflow, as a function of the capacity factor?

Routes T tokens to N experts with (a) uniform random routing and (b) a skewed
router that favors expert 0, then applies Equation 3 and counts the dropped
tokens. The paper reports dropped-token rates of "typically < 1%" with the
auxiliary loss on (Section 2.2); the uniform case shows why a capacity factor
above 1.0 is needed even when the router is perfectly balanced on average.
"""

import math

import numpy as np

T, N = 1024, 8
rng = np.random.default_rng(0)
trials = 200


def drop_rate(probs, cf):
    capacity = math.ceil(T / N * cf)
    dropped = 0
    for _ in range(trials):
        idx = rng.choice(N, size=T, p=probs)
        counts = np.bincount(idx, minlength=N)
        dropped += np.maximum(counts - capacity, 0).sum()
    return dropped / (trials * T)


uniform = np.full(N, 1 / N)
skewed = np.array([0.30] + [0.70 / (N - 1)] * (N - 1))
print(f"T={T} tokens, N={N} experts, {trials} random batches")
print("capacity factor | capacity | dropped (uniform) | dropped (skewed, 30% to expert 0)")
for cf in (1.0, 1.25, 1.5, 2.0):
    cap = math.ceil(T / N * cf)
    print(f"{cf:>15} | {cap:>8} | {drop_rate(uniform, cf):>17.4f} | {drop_rate(skewed, cf):>.4f}")
