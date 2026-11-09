"""Compute the numbers for the forward algorithm, the gradient, and the scaling examples.

Run: python3 scripts/forward_example.py
"""

import numpy as np

from ctc_ref import BLANK, backward, brute_force, extend, forward, grad_u, prob

np.set_printoptions(precision=4, suppress=True, linewidth=120)

y = np.array([[0.4, 0.5, 0.1], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])
label = (1, 2)  # "ab"
print("l' =", extend(label), "(0 is blank, 1 is a, 2 is b); |l'| =", len(extend(label)))

a = forward(y, label)
print("alpha (rows t = 1..3, columns s = 1..5)")
print(a)
print(f"p(l|x) from Equation 8 = alpha_3(5) + alpha_3(4) = {a[2, 4]:.4f} + {a[2, 3]:.4f} = {prob(y, label):.4f}")
print(f"p(l|x) by brute force  = {brute_force(y, label)[0]:.4f}")

b = backward(y, label)
print()
print("beta")
print(b)
print("Equation 14 at each t: sum_s alpha beta / y_{l'_s} =")
lp = extend(label)
for t in range(3):
    print(f"  t={t + 1}: {sum(a[t, s] * b[t, s] / y[t, lp[s]] for s in range(5)):.4f}")

print()
print("The repeat rule: label 'aa' needs a blank between the two a's")
print(f"p(aa|x) = {prob(y, (1, 1)):.4f}; alpha for l' = [-, a, -, a, -]:")
print(forward(y, (1, 1)))

print()
print("Gradient (Equation 16) for label 'ab', and a finite-difference check through the softmax")
u = np.log(y)  # any u with softmax(u) = y works


def softmax(u):
    e = np.exp(u - u.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def loss(u):
    return -np.log(prob(softmax(u), label))


g = grad_u(y, label)
num = np.zeros_like(u)
for i in np.ndindex(*u.shape):
    d = np.zeros_like(u)
    d[i] = 1e-6
    num[i] = (loss(u + d) - loss(u - d)) / 2e-6
print(g)
print("max |analytic - numeric| < 1e-8:", bool(np.abs(g - num).max() < 1e-8))
print("each row sums to", np.round(g.sum(axis=1), 10).tolist())

print()
print("Counting paths: set every y to 1, and the forward pass counts the paths of B^-1(l)")
for T, U in [(3, 2), (10, 3), (50, 5), (200, 20)]:
    # Distinct labels 1..U, so the repeat rule never applies.
    yy = np.ones((T, U + 1))
    print(f"  T={T:>3}, U={U:>2}: {forward(yy, tuple(range(1, U + 1)))[-1, -2:].sum():.4g} paths")

print()
print("Underflow: T = 400, y = 0.05 for every frame and every symbol")
T, K = 400, 20
yy = np.full((T, K), 1.0 / K)
lab = (1, 2, 3, 4, 5)
a = forward(yy, lab)
print("unscaled p(l|x) =", a[-1, -1] + a[-1, -2], "(float64 underflows to 0, so ln p = -inf)")


def log_prob_scaled(y, label):
    """Rescaled forward pass. It zeroes the states that can't finish in the time left
    (the note after Equation 7), so that ln p(l|x) = sum_t ln C_t holds exactly."""
    lp = extend(label)
    S = len(lp)
    T = y.shape[0]
    total = 0.0
    a = np.zeros(S)
    for t in range(T):
        new = np.zeros(S)
        if t == 0:
            new[0] = y[0, BLANK]
            new[1] = y[0, lp[1]]
        else:
            for s in range(S):
                tot = a[s] + (a[s - 1] if s >= 1 else 0.0)
                if s >= 2 and lp[s] != BLANK and lp[s] != lp[s - 2]:
                    tot += a[s - 2]
                new[s] = tot * y[t, lp[s]]
        new[: max(0, S - 2 * (T - 1 - t) - 2)] = 0.0
        C = new.sum()
        a = new / C
        total += np.log(C)
    return total


print(f"rescaled ln p(l|x) = sum_t ln C_t = {log_prob_scaled(yy, lab):.4f}")
# Check the rescaled routine against the unscaled one on the small example.
print("rescaled vs unscaled on the T=3 example:",
      f"{log_prob_scaled(np.array([[0.4, 0.5, 0.1], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]), (1, 2)):.4f}",
      f"{np.log(0.273):.4f}")
