"""Equation 3 on a group of two outputs with different lengths."""

import numpy as np

from grpo_lib import group_advantages, grpo_objective, kl_k3, token_terms

eps, beta = 0.2, 0.04
rewards = np.array([1.0, 0.0])
adv = group_advantages(rewards)
print("advantages", np.round(adv, 4))

ratio = np.array([[1.3, 0.9, 1.0, 1.0], [1.1, 1.0, 0.7, 1.25]])
mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
logp_old = np.full((2, 4), -1.0)
logp_new = logp_old + np.log(ratio)
d = np.array([[0.1, -0.2, 0.0, 0.0], [0.0, 0.3, -0.1, 0.2]])  # log pi_ref - log pi
logp_ref = logp_new + d

clip_part = token_terms(logp_new, logp_old, adv, eps)
kl = kl_k3(logp_new, logp_ref)
per_token = clip_part - beta * kl
print("clipped term per token")
print(np.round(clip_part, 4))
print("k3 per token")
print(np.round(kl, 4))
per_output = (per_token * mask).sum(1) / mask.sum(1)
print("per-output mean over its own tokens", np.round(per_output, 4))
J = grpo_objective(logp_new, logp_old, logp_ref, adv, mask, eps, beta)
print("J (mean over outputs)             ", round(float(J), 4))
global_mean = (per_token * mask).sum() / mask.sum()
print("global token mean (not the paper) ", round(float(global_mean), 4))
print("beta = 0 J                        ", round(float(grpo_objective(logp_new, logp_old, logp_ref, adv, mask, eps, 0.0)), 4))
