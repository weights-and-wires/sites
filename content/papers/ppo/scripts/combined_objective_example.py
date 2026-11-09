#!/usr/bin/env python3
"""The combined objective L^{CLIP+VF+S} (Section 5, Equation 9) on a batch of 3."""

import numpy as np
from ppo_lib import l_clip

eps, c1, c2 = 0.2, 0.5, 0.01
probs_old = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.1, 0.1, 0.8]])
probs = np.array([[0.65, 0.2, 0.15], [0.3, 0.5, 0.2], [0.1, 0.1, 0.8]])
actions = np.array([0, 1, 2])
adv = np.array([1.0, -0.5, 2.0])
values = np.array([1.0, 0.0, 2.5])
returns = np.array([1.5, 0.5, 2.0])

idx = np.arange(3)
ratio = probs[idx, actions] / probs_old[idx, actions]
clip_t = l_clip(ratio, adv, eps)
vf_t = (values - returns) ** 2
ent_t = -np.sum(probs * np.log(probs), axis=1)
print("ratio     :", ratio.round(4))
print("L_CLIP_t  :", clip_t.round(4), "mean", clip_t.mean().round(4))
print("L_VF_t    :", vf_t.round(4), "mean", vf_t.mean().round(4))
print("S_t (nats):", ent_t.round(4), "mean", ent_t.mean().round(4))
obj = clip_t.mean() - c1 * vf_t.mean() + c2 * ent_t.mean()
print(f"c1={c1}, c2={c2}")
print("objective (maximize):", obj.round(4))
print("loss (minimize)     :", (-obj).round(4))
print("uniform entropy ln 3:", np.log(3).round(4))
