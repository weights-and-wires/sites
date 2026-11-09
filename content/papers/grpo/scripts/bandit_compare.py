"""GRPO against a policy gradient with a learned value baseline, on a Bernoulli bandit.

Both methods use one update per batch, as in the paper's setup (Section 4.2),
so the ratio is 1 and the clip never triggers. The only difference is the baseline.

  GRPO:   A_i = (r_i - mean(group)) / std(group), no learned parameters.
  Critic: A_i = r_i - V, where V is a learned scalar updated toward the batch mean
          (the bandit version of PPO's value model; no state, so V is one number).
  Group mean: A_i = r_i - mean(group), with no division by the std, to separate the
          effect of the group baseline from the effect of the std rescaling.
"""

import numpy as np

from grpo_lib import grpo_update, softmax

P = np.array([0.2, 0.5, 0.8])  # success probability of each arm
G, STEPS, LR, BETA, SEEDS = 8, 200, 0.5, 0.0, 50
CHECK = [10, 25, 50, 100, 200]


def run(method, seed):
    rng = np.random.default_rng(seed)
    logits = np.zeros(3)
    ref = np.zeros(3)
    v = 0.0
    out = []
    for step in range(1, STEPS + 1):
        pi = softmax(logits)
        a = rng.choice(3, size=G, p=pi)
        r = (rng.random(G) < P[a]).astype(float)
        if method == "grpo":
            logits = grpo_update(logits, a, r, ref, LR, BETA)
        elif method == "group_mean":
            adv = r - r.mean()
            grad = (adv[:, None] * (np.eye(3)[a] - pi)).mean(axis=0)
            logits = logits + LR * grad
        else:
            adv = r - v
            grad = (adv[:, None] * (np.eye(3)[a] - pi)).mean(axis=0)
            logits = logits + LR * grad
            v += 0.2 * (r.mean() - v)
        if step in CHECK:
            out.append(softmax(logits)[2])
    return out


for method in ["grpo", "group_mean", "critic"]:
    res = np.array([run(method, s) for s in range(SEEDS)])
    print(method)
    for j, c in enumerate(CHECK):
        print(f"  step {c:>3}: mean P(best arm) {res[:, j].mean():.3f}, std over seeds {res[:, j].std():.3f}")
