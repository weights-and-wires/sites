#!/usr/bin/env python3
"""How much does Adam shrink a weight because of the L2 term, and how much with decoupled decay?

Run from anywhere:  python3 content/papers/adamw/scripts/adam_effective_decay.py
"""
import numpy as np

alpha, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

print("Part 1: the first Adam step when the loss gradient is zero, theta = 1")
for lam in (0.1, 0.01, 0.001):
    g = lam * 1.0                                   # Algorithm 2 line 6 with grad f = 0
    m_hat = (1 - beta1) * g / (1 - beta1)
    v_hat = (1 - beta2) * g * g / (1 - beta2)
    step = alpha * m_hat / (np.sqrt(v_hat) + eps)   # line 12, adaptive part
    print(f"  L2 lambda'={lam:<6}: theta shrinks by {step:.6f}   (AdamW lambda={lam}: shrinks by {lam:.6f})")

print("Part 2: per-step relative shrinkage of the L2 term when the loss gradient has scale s")
print("  g_t = s * N(0,1) + lambda' * theta with theta held at 1, lambda' = 0.1, 2000 steps, seed 0")
lam_l2 = 0.1
for s in (0.1, 1.0, 10.0):
    rng = np.random.default_rng(0)
    m = v = 0.0
    rates = []
    for t in range(1, 2001):
        g = s * rng.standard_normal() + lam_l2 * 1.0
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        v_hat = v / (1 - beta2**t)
        rates.append(alpha * lam_l2 / (np.sqrt(v_hat) + eps))   # the lambda' * theta part of the step
    print(f"  s={s:<5}: mean L2 shrinkage per step = {np.mean(rates[200:]):.3e}")
print(f"  AdamW with lambda = 1e-4 shrinks by {1e-4:.3e} per step for every s")
