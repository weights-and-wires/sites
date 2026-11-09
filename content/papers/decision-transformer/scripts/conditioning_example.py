#!/usr/bin/env python3
"""Why conditioning on return-to-go improves a policy trained on random data.

A 7-cell corridor (cells 0..6), start in cell 3, goal in cell 6. Action 0 moves
left, action 1 moves right; the left wall at cell 0 blocks. Every step gives
reward -1 and the episode ends at the goal, so the return from a state equals
minus the number of steps still to take. This mirrors the shortest-path
example in Section 1 and Appendix A.3.

The dataset is random-walk trajectories only. A count table stands in for the
model: pi(a | s, R) = count(s, R, a) / count(s, R). It is a one-timestep
version of the idea, not a transformer.
"""

import numpy as np

N_STATES, GOAL, START = 7, 6, 3
R_MIN = -40  # returns below this share one bucket


def step(s, a):
    s2 = min(s + 1, GOAL) if a == 1 else max(s - 1, 0)
    return s2, -1.0, s2 == GOAL


def random_walks(n, rng):
    S, A, G = [], [], []
    for _ in range(n):
        s, traj, done = START, [], False
        while not done:
            a = int(rng.integers(2))
            s2, r, done = step(s, a)
            traj.append((s, a, r))
            s = s2
        rtg = np.cumsum([r for _, _, r in traj][::-1])[::-1]
        for (s_, a_, _), g in zip(traj, rtg):
            S.append(s_), A.append(a_), G.append(g)
    return np.array(S), np.array(A), np.array(G)


def fit_table(S, A, G):
    table = np.zeros((N_STATES, -R_MIN + 1, 2))
    np.add.at(table, (S, np.clip(G, R_MIN, 0).astype(int) - R_MIN, A), 1)
    return table


def probs(table, s, R):
    c = table[s, int(np.clip(R, R_MIN, 0)) - R_MIN]
    return c / c.sum() if c.sum() > 0 else np.array([0.5, 0.5])


def rollout(policy, target, rng, max_steps=200):
    s, R, steps, done = START, target, 0, False
    while not done and steps < max_steps:
        a = int(rng.choice(2, p=policy(s, R)))
        s, r, done = step(s, a)
        R -= r
        steps += 1
    return steps, done


rng = np.random.default_rng(0)
S, A, G = random_walks(2000, rng)
table = fit_table(S, A, G)
print("dataset: 2000 random walks,", len(S), "timesteps")
print("mean return-to-go over timesteps in state 3:", round(float(np.mean(G[(S == START)])), 2))

c3 = table[START].sum(-1)
print("\nAt state 3, count of dataset timesteps by return-to-go:")
for R in (-3, -5, -7, -9):
    cnt = table[START, R - R_MIN]
    print(f"  R={R}: left={int(cnt[0])}, right={int(cnt[1])}, P(right)={cnt[1] / cnt.sum():.3f}")
marg = table[START].sum(0)
print(f"marginal over all R (behavior cloning): left={int(marg[0])}, right={int(marg[1])}, P(right)={marg[1] / marg.sum():.3f}")
print("R=-2 at state 3 (impossible, no data): count =", int(table[START, -2 - R_MIN].sum()))

bc = lambda s, R: marg_s[s]
marg_s = {s: table[s].sum(0) / table[s].sum() for s in range(N_STATES - 1)}
dt = lambda s, R: probs(table, s, R)

print("\nRollouts from state 3, 500 episodes each:")
for name, pol, target in (("behavior cloning", bc, None), ("conditioned R=-3", dt, -3.0),
                          ("conditioned R=-5", dt, -5.0), ("conditioned R=-9", dt, -9.0)):
    lens = []
    for _ in range(500):
        n, done = rollout(pol, target if target is not None else 0.0, rng)
        lens.append(n)
    print(f"  {name:<18}: mean steps {np.mean(lens):6.2f}, min {min(lens)}")
