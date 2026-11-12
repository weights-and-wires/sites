"""Compute the numbers for Theorem 1: the natural gradient equals the compatible weights.

The policy is a softmax over K actions with logits theta . phi[s, a], so
psi(s, a) = grad_theta log pi(a; s, theta) = phi[s, a] - sum_b pi(b; s) phi[s, b]
(Equation 5). The state distribution rho and the action values Q are fixed arrays,
so Equation 1 and the fit in Section 3.1 both use the same rho and Q.

Run: python3 compatible_weights.py
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
psi = phi - np.einsum("sk,skd->sd", pi, phi)[:, None, :]  # shape (S, K, d)

# Equation 1: grad eta = sum_{s,a} rho(s) grad pi(a; s) Q(s, a), with grad pi = pi * psi.
grad = np.einsum("s,sk,skd,sk->d", rho, pi, psi, Q)

# Equations 2 and 3: F = sum_s rho(s) E_pi[psi psi^T].
F = np.einsum("s,sk,ska,skb->ab", rho, pi, psi, psi)

natural = np.linalg.solve(F, grad)  # Equation 4

# Section 3.1: w minimizes sum_{s,a} rho(s) pi(a; s) (psi . w - Q)^2.
weights = (rho[:, None] * pi).reshape(-1)
X = psi.reshape(S * K, d)
y = Q.reshape(-1)
sw = np.sqrt(weights)
w, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)

print(f"rho                          = {rho}")
print(f"pi (rows are states) =\n{pi}")
print(f"grad eta (Equation 1)        = {grad}")
print(f"F =\n{F}")
print(f"eigenvalues of F             = {np.linalg.eigvalsh(F)}")
print(f"natural gradient F^-1 grad   = {natural}")
print(f"least-squares w              = {w}")
print(f"max |w - F^-1 grad|          = {np.abs(w - natural).max():.2e}")

# The fit residual does not need to be zero. Theorem 1 holds for the best fit.
resid = X @ w - y
print(f"weighted squared error at w  = {np.sum(weights * resid**2):.4f}")
print(f"weighted squared error of Q  = {np.sum(weights * y**2):.4f}  (w = 0)")

# An unweighted fit gives a different answer, so rho(s) pi(a; s) matters.
w_unw, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"unweighted least squares     = {w_unw}")

# Adding a constant per state to Q does not change w, because sum_a pi psi = 0.
Q_shift = Q + np.array([10.0, -3.0, 7.0])[:, None]
w_shift, *_ = np.linalg.lstsq(X * sw[:, None], Q_shift.reshape(-1) * sw, rcond=None)
print(f"w after Q += per-state const = {w_shift}")
print(f"max |w_shift - w|            = {np.abs(w_shift - w).max():.2e}")

# The gradient of eta also ignores the constant, so the natural gradient does too.
grad_shift = np.einsum("s,sk,skd,sk->d", rho, pi, psi, Q_shift)
print(f"max |grad_shift - grad|       = {np.abs(grad_shift - grad).max():.2e}")
