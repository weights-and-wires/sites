#!/usr/bin/env python3
"""Numbers for the clipped surrogate objective (Section 3, Equations 6 and 7)."""

import numpy as np
from ppo_lib import clip_grad_logp, l_clip

eps = 0.2
print(f"eps = {eps}; clip range = [{1 - eps:.1f}, {1 + eps:.1f}]")
print("single-sample table: A, r, r*A, L_CLIP, dL/dr")
for a, r in [(1, 1.1), (1, 1.5), (1, 0.5), (-1, 0.9), (-1, 0.5), (-1, 1.5)]:
    term = float(l_clip(r, a, eps))
    grad_r = a if r * a <= np.clip(r, 1 - eps, 1 + eps) * a else 0.0
    print(f"A={a:+d} r={r:.1f} r*A={r * a:+.2f} L_CLIP={term:+.2f} dL/dr={grad_r:+.0f}")

print()
ratio = np.array([1.3, 0.7, 1.1, 0.5])
adv = np.array([2.0, 1.0, -1.0, -2.0])
print("batch ratios:", ratio, "advantages:", adv)
print("terms r*A     :", ratio * adv, "mean", np.mean(ratio * adv).round(4))
print("terms L_CLIP  :", l_clip(ratio, adv, eps).round(4), "mean", np.mean(l_clip(ratio, adv, eps)).round(4))
logp_old = np.log(np.array([0.25, 0.4, 0.5, 0.3]))
logp_new = logp_old + np.log(ratio)
g = clip_grad_logp(logp_new, logp_old, adv, eps)
print("dL/dlogp_new  :", g.round(4))
