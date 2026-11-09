"""Compute the numbers for the GAE examples on the page.

Notation follows the paper (Section 3): reward r_t, value estimate V, TD residual
delta_t^V (Equation 11 area), k-step estimators A_t^(k) (Equation 14), and
GAE(gamma, lambda) (Equation 16).

Run: python3 scripts/gae_example.py
"""

import numpy as np

np.set_printoptions(precision=5, suppress=True)

gamma = 0.9
r = np.array([1.0, 0.0, 0.0, 2.0])  # r_0 .. r_3; the episode ends after step 3
V = np.array([0.5, 0.4, 0.6, 0.3])  # V(s_0) .. V(s_3); V of the terminal state is 0
T = len(r)
V_next = np.append(V[1:], 0.0)

delta = r + gamma * V_next - V
print("TD residuals delta_t        ", delta)


def k_step(t, k):
    """A_t^(k) = sum_{l<k} gamma^l delta_{t+l}, with delta = 0 after the episode ends (Equation 14)."""
    end = min(t + k, T)
    return sum(gamma ** (i - t) * delta[i] for i in range(t, end))


def gae(lam):
    """Equation 16, computed by the backward recursion A_t = delta_t + gamma lambda A_{t+1}."""
    A = np.zeros(T)
    nxt = 0.0
    for t in reversed(range(T)):
        nxt = delta[t] + gamma * lam * nxt
        A[t] = nxt
    return A


def gae_direct(lam):
    """Equation 16 written as a direct sum over l."""
    return np.array([sum((gamma * lam) ** l * delta[t + l] for l in range(T - t)) for t in range(T)])


print()
print("k-step estimators at t = 0 (Equation 14)")
for k in range(1, T + 1):
    print(f"  A_0^({k})                  {k_step(0, k):.5f}")

returns = np.array([sum(gamma**l * r[t + l] for l in range(T - t)) for t in range(T)])
print("discounted return from t    ", returns)
print("return minus V(s_t)         ", returns - V)

print()
for lam in (0.0, 0.5, 1.0):
    A = gae(lam)
    assert np.allclose(A, gae_direct(lam))
    print(f"GAE(gamma=0.9, lambda={lam})  ", A)
assert np.allclose(gae(0.0), delta)  # Equation 17
assert np.allclose(gae(1.0), returns - V)  # Equation 18

lam = 0.5
w = np.array([(1 - lam) * lam ** (k - 1) for k in range(1, T)])  # weights on A^(1) .. A^(T-1)
tail = lam ** (T - 1)  # the remaining weight goes on A^(T), the full return minus V
mix = sum(w[k - 1] * k_step(0, k) for k in range(1, T)) + tail * k_step(0, T)
print()
print("weights (1 - lambda) lambda^(k-1), k = 1..3:", w)
print(f"weight left for the full return A_0^(4):     {tail:.5f}")
print(f"weighted mixture of k-step estimators at t=0: {mix:.5f}  (Equation 16 gives {gae(lam)[0]:.5f})")
assert np.isclose(mix, gae(lam)[0])

# Backward recursion for lambda = 0.5, step by step.
A = gae(0.5)
print()
print("backward recursion, gamma*lambda = 0.45")
print(f"  A_3 = delta_3                       = {delta[3]:.5f}")
for t in (2, 1, 0):
    print(f"  A_{t} = delta_{t} + 0.45 * A_{t + 1}  = {delta[t]:.5f} + 0.45 * {A[t + 1]:.5f} = {A[t]:.5f}")

# A value function that overfits the batch: V(s_t) equals the discounted return from t (Section 6.1).
V_overfit = returns
delta_o = r + gamma * np.append(V_overfit[1:], 0.0) - V_overfit
print()
print("V fit exactly to this batch's returns:", V_overfit)
print("  TD residuals                       ", np.round(delta_o, 10) + 0.0)
for lam in (0.0, 0.5, 1.0):
    A_o = np.zeros(T)
    nxt = 0.0
    for t in reversed(range(T)):
        nxt = delta_o[t] + gamma * lam * nxt
        A_o[t] = nxt
    print(f"  GAE(gamma=0.9, lambda={lam})         ", np.round(A_o, 10) + 0.0)
