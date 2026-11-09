"""Replay versus online updates on the 5x5 gridworld (the "experience replay" example).

Two arms use the same network, the same epsilon schedule, and the same number
of environment steps. The replay arm samples a minibatch of 32 from a memory of
2,000 transitions at every step. The online arm updates on the one transition it
just saw, which is the update that Section 2 calls "the familiar Q-learning
algorithm". The arms differ in two ways at once: decorrelation and data reuse
(32 samples per step against 1). Each arm gets a learning-rate sweep.

Run: python3 scripts/ablation.py     (about 40 seconds)
"""

import warnings

import numpy as np

import gridworld as g

warnings.simplefilter("ignore")  # the online arm can overflow at high learning rates

SEEDS = 20
LRS = [0.05, 0.02, 0.01, 0.005, 0.002]

print(f"{SEEDS} seeds, 4000 steps each. Cell = seeds whose greedy policy reaches the goal.")
print(f"{'learning rate':>14} {'replay':>8} {'online':>8}")
best = {"replay": 0, "online": 0}
for lr in LRS:
    row = {}
    for name, replay in (("replay", True), ("online", False)):
        wins = sum(g.greedy_success(g.train(seed, replay=replay, lr=lr)[0]) for seed in range(SEEDS))
        row[name] = wins
        best[name] = max(best[name], wins)
    print(f"{lr:>14} {row['replay']:>6}/{SEEDS} {row['online']:>6}/{SEEDS}")
print(f"best over the sweep: replay {best['replay']}/{SEEDS}, online {best['online']}/{SEEDS}")

# Correlation of consecutive samples. Collect 2,000 transitions from an
# epsilon = 0.3 policy on the trained network, then compare the lag-1
# correlation of the row coordinate for consecutive transitions with the
# correlation of two transitions drawn uniformly from the same memory.
rng = np.random.default_rng(0)
p, _, _ = g.train(0, replay=True)
pos, xs = g.reset(), []
for t in range(2000):
    s = g.features(pos)
    a = int(rng.integers(4)) if rng.random() < 0.3 else int(np.argmax(g.q_values(p, s)))
    pos, _, done = g.step(pos, a)
    xs.append(s[0])
    if done:
        pos = g.reset()
xs = np.array(xs)
consecutive = np.corrcoef(xs[:-1], xs[1:])[0, 1]
i = rng.integers(len(xs), size=100_000)
j = rng.integers(len(xs), size=100_000)
uniform = np.corrcoef(xs[i], xs[j])[0, 1]
print(f"\ncorrelation of the row coordinate: consecutive samples = {consecutive:.2f}, uniform pairs = {uniform:.2f}")
