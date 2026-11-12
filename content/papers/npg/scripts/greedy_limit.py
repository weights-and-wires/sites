"""Compute the numbers for Theorem 2: a large natural gradient step becomes greedy.

The policy is in the exponential family, pi(a; s, theta) proportional to exp(theta . phi[s, a]).
A step theta + alpha * direction changes the logits by alpha * direction . phi[s, a], so for
a large alpha the action with the largest value of direction . phi[s, a] takes all the
probability. This script compares the natural gradient direction (Theorem 2) with the plain
gradient direction, for the same fixed rho and Q as `compatible_weights.py`.

Run: python3 greedy_limit.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

rng = np.random.default_rng(0)
S, K, d = 3, 3, 3
phi = rng.normal(size=(S, K, d))
theta = rng.normal(size=d)
rho = np.array([0.5, 0.3, 0.2])
Q = rng.normal(size=(S, K))


def probs(theta):
    z = phi @ theta
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


pi = probs(theta)
psi = phi - np.einsum("sk,skd->sd", pi, phi)[:, None, :]
grad = np.einsum("s,sk,skd,sk->d", rho, pi, psi, Q)
F = np.einsum("s,sk,ska,skb->ab", rho, pi, psi, psi)
natural = np.linalg.solve(F, grad)

f_nat = psi @ natural  # f(s, a; w) = w . psi(s, a), with w the natural gradient
f_plain = psi @ grad  # the same score for the plain gradient direction

print("Greedy action under the compatible approximator f(s, a; w) = w . psi(s, a):")
print(f"  argmax_a f per state             = {f_nat.argmax(axis=1)}")
print(f"  argmax_a Q per state             = {Q.argmax(axis=1)}")
print(f"  argmax_a (grad . phi) per state  = {(phi @ grad).argmax(axis=1)}  (plain gradient)")
print(f"  argmax_a (w . phi) per state     = {(phi @ natural).argmax(axis=1)}  (natural gradient)")
print(f"  f for the natural gradient =\n{f_nat}")
print(f"  mean over pi of f, per state     = {np.sum(pi * f_nat, axis=1)}  (always 0)")

print("\nPolicy after a step of size alpha along each direction:")
for alpha in (0.0, 1.0, 10.0, 1000.0):
    p_nat = probs(theta + alpha * natural)
    print(f"  alpha = {alpha:g}")
    print(f"    natural =\n{p_nat}")
    p_plain = probs(theta + alpha * grad)
    print(f"    plain =\n{p_plain}")

print("\nFor the plain gradient, a large step picks the action with the largest")
print("grad . phi, which need not be the action with the largest f(s, a; w):")
for s in range(S):
    best_plain = int((phi[s] @ grad).argmax())
    best_f = int(f_nat[s].argmax())
    print(f"  state {s}: plain picks {best_plain}, f picks {best_f}, "
          f"f({best_plain}) = {f_nat[s, best_plain]:.4f}, max f = {f_nat[s].max():.4f}")
