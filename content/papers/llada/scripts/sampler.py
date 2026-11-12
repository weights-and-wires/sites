"""Compute the numbers for the sampler examples.

Run: python3 scripts/sampler.py
"""

import numpy as np

MASK = 5
V = 5  # tokens 0..4; the mask id is V
TARGET = np.array([2, 0, 3, 1, 2, 0])
BASE = np.array([0.9, 0.4, 0.9, 0.4, 0.9, 0.4])
L = len(TARGET)


def toy_logits(x):
    """Deterministic toy predictor. Position i puts probability c on TARGET[i] and 1 - c on a
    wrong token, where c = min(0.95, BASE[i] + 0.15 * the number of unmasked neighbors)."""
    logits = np.full((L, V), np.log(1e-9))
    for i in range(L):
        c = min(0.95, BASE[i] + 0.15 * sum(1 for j in (i - 1, i + 1) if 0 <= j < L and x[j] != MASK))
        logits[i, TARGET[i]] = np.log(c)
        logits[i, (TARGET[i] + 1) % V] = np.log(1 - c)
    return logits


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def step(x, n_un, strategy, rng):
    pred = toy_logits(x).argmax(axis=-1)
    conf = softmax(toy_logits(x)).max(axis=-1)
    masked = x == MASK
    if strategy == "low_confidence":
        conf = np.where(masked, conf, np.inf)
        keep = np.argsort(-conf, kind="stable")[:n_un]
    else:
        fresh = rng.permutation(np.flatnonzero(masked))[: n_un - int((~masked).sum())]
        keep = np.concatenate([np.flatnonzero(~masked), fresh])
    out = np.full(L, MASK)
    out[keep] = np.where(masked, pred, x)[keep]
    return out


def sample(strategy, N, rng=None):
    x = np.full(L, MASK)
    trace = []
    for k in range(1, N + 1):
        x = step(x, L * k // N, strategy, rng)
        trace.append(x.tolist())
    return x, trace


print("target =", TARGET.tolist(), "(5 marks M)")
print("n_un per step for N=3:", [L * k // 3 for k in range(1, 4)])
x, trace = sample("low_confidence", 3)
for k, row in enumerate(trace, 1):
    print(f"low-confidence step {k}: {row}")
print("low-confidence correct:", bool((x == TARGET).all()))

accs = []
for seed in range(2000):
    x, _ = sample("random", 3, np.random.default_rng(seed))
    accs.append(float((x == TARGET).mean()))
print(f"random remasking over 2000 seeds: mean token accuracy={np.mean(accs):.4f}"
      f"  fraction fully correct={np.mean([a == 1.0 for a in accs]):.4f}")
x, _ = sample("random", 3, np.random.default_rng(0))
print("random remasking, seed 0 result:", x.tolist())

x, _ = sample("low_confidence", 1)
print("one step (N=1):", x.tolist(), "token accuracy", float((x == TARGET).mean()))
