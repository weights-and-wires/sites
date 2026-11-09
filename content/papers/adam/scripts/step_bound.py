"""The effective step size and its bounds (Section 2.1), and invariance to
gradient rescaling.

Run: python3 scripts/step_bound.py
"""

import numpy as np

alpha, beta1, beta2 = 1e-3, 0.9, 0.999


def steps(grads, eps=1e-8):
    m = v = 0.0
    out = []
    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        out.append(alpha * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps))
    return np.array(out)


bound = alpha * (1 - beta1) / np.sqrt(1 - beta2)
print(f"bound alpha*(1-beta1)/sqrt(1-beta2) = {bound / alpha:.4f} * alpha")

for zeros in (20, 5000):
    sparse = steps([0.0] * zeros + [1.0])
    print(f"sparse: gradient 1 only at t={zeros + 1}, step/alpha = {sparse[-1] / alpha:.4f}")
first = steps([5.0])
print(f"first step with g=5: step/alpha = {first[0] / alpha:.4f}")

noisy = steps(np.random.default_rng(0).normal(size=2000))
print(f"noisy gradients: max |step|/alpha = {np.abs(noisy).max() / alpha:.4f}")

g = np.random.default_rng(1).normal(size=200)
a = steps(g, eps=0.0)
b = steps(1000.0 * g, eps=0.0)
print(f"rescale gradients by 1000 (eps=0): max |difference| = {np.abs(a - b).max():.2e}")
