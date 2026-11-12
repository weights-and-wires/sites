"""Numbers for the "Reparameterized policy gradient" example (Section 4.2 Equations 10 to 13).

Toy problem at one state: a ~ N(mu, sigma^2), Q(a) = -(a - 2)^2. Then E[Q] = -(mu - 2)^2 - sigma^2,
so the true gradient of E[Q] with respect to mu is -2 (mu - 2).

Run: python3 scripts/reparam_vs_score.py
"""

import numpy as np

mu, sigma, batch, trials = 0.5, 1.0, 16, 20000
true_grad = -2 * (mu - 2)
rng = np.random.default_rng(0)


def q(a):
    return -((a - 2.0) ** 2)


def dq(a):
    return -2.0 * (a - 2.0)


eps = rng.standard_normal((trials, batch))
a = mu + sigma * eps

reparam = dq(a).mean(axis=1)                        # d/dmu E[Q(mu + sigma eps)] = E[Q'(a)]
score = (q(a) * (a - mu) / sigma**2).mean(axis=1)   # E[Q(a) d/dmu log N(a; mu, sigma)]

print(f"true gradient of E[Q] w.r.t. mu = {true_grad:.3f}")
print(f"batch of {batch}, {trials} trials")
print(f"  reparameterized: mean = {reparam.mean():.3f}  variance = {reparam.var():.3f}  (theory 4 sigma^2 / {batch} = {4 * sigma**2 / batch:.3f})")
print(f"  score function:  mean = {score.mean():.3f}  variance = {score.var():.3f}")
print(f"  variance ratio score / reparameterized = {score.var() / reparam.var():.1f}")

# Same comparison with a baseline of E[Q] subtracted from the score estimator.
base = -((mu - 2) ** 2) - sigma**2
score_b = ((q(a) - base) * (a - mu) / sigma**2).mean(axis=1)
print(f"  score with the exact mean as baseline: variance = {score_b.var():.3f}")

# A wider policy makes the score estimator noisier still; the reparameterized one grows slowly.
for s in (0.25, 1.0, 2.0):
    a = mu + s * rng.standard_normal((trials, batch))
    rp = dq(a).mean(axis=1)
    sc = (q(a) * (a - mu) / s**2).mean(axis=1)
    print(f"  sigma = {s}: reparam variance = {rp.var():.3f}  score variance = {sc.var():.3f}")
