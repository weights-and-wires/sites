#!/usr/bin/env python3
"""One PPO iteration on a 3-armed bandit (Section 5, Algorithm 1, with N = 1 actor).

The policy is a softmax over three logits. The batch is sampled once from
pi_old, and then K epochs of gradient ascent reuse it. Compare the clipped
objective (Equation 7) with the unclipped one (Equation 6).
"""

import numpy as np
from ppo_lib import kl_categorical, ppo_epochs, softmax

rng = np.random.default_rng(0)
means = np.array([0.0, 0.5, 1.0])
theta_old = np.zeros(3)
pi_old = softmax(theta_old)
n = 64
actions = rng.choice(3, size=n, p=pi_old)
rewards = means[actions] + 0.3 * rng.normal(size=n)
adv = rewards - rewards.mean()  # batch-mean baseline, not GAE
print("pi_old:", pi_old.round(4), "batch size:", n)
print("per-arm counts:", np.bincount(actions, minlength=3))
print("per-arm mean advantage:", np.array([adv[actions == a].mean() for a in range(3)]).round(4))

lr = 0.5
print(f"\nlr = {lr}; ratio is pi_new(a)/pi_old(a) for the arm; KL is KL[pi_old, pi_new]")
print("eps    epochs  pi_new(best)  ratio(best)  KL")
for eps in [None, 0.2]:
    for epochs in [1, 10, 50, 200]:
        theta = ppo_epochs(theta_old, actions, adv, eps, lr, epochs)
        pi = softmax(theta)
        label = "none" if eps is None else f"{eps:.1f} "
        print(f"{label}   {epochs:5d}   {pi[2]:.4f}        {pi[2] / pi_old[2]:.4f}       {kl_categorical(pi_old, pi):.4f}")
