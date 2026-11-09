"""Compute the numbers for the Fisher matrix, conjugate gradient, step size, and line search examples.

The policy is a softmax over K actions with logits phi(s, a) . theta, evaluated on
N sampled states (a contextual bandit, so the surrogate is L(theta) = mean_s sum_a pi(a|s) A(s, a)).
Notation follows Appendix C of the paper: g is the gradient of L, A (here F) is the
Fisher matrix, s is the search direction, beta is the step length.

Run: python3 scripts/natural_gradient_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---- 1. Conjugate gradient on a 2x2 system, small enough to follow by hand ----
F2 = np.array([[4.0, 1.0], [1.0, 3.0]])
g2 = np.array([1.0, 2.0])


def conjugate_gradient(fvp, b, iters, tol=1e-10):
    x = np.zeros_like(b)
    r = b.copy()
    p = b.copy()
    rs = r @ r
    for _ in range(iters):
        Ap = fvp(p)
        alpha = rs / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if rs_new < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


print("== Conjugate gradient on A = [[4, 1], [1, 3]], g = [1, 2]")
x1 = conjugate_gradient(lambda v: F2 @ v, g2, 1)
x2 = conjugate_gradient(lambda v: F2 @ v, g2, 2)
print("after 1 iteration        =", x1)
print("after 2 iterations       =", x2)
print("np.linalg.solve          =", np.linalg.solve(F2, g2))
print("residual after 2         =", g2 - F2 @ x2)
delta = 0.01
beta = np.sqrt(2 * delta / (x2 @ F2 @ x2))
print("s^T A s                  =", round(x2 @ F2 @ x2, 6))
print("beta = sqrt(2 delta/s^T A s), delta = 0.01 =", round(beta, 6))
print("quadratic KL of beta*s   =", round(0.5 * beta**2 * (x2 @ F2 @ x2), 6))
print("2 * delta                =", 2 * delta)
print("2 * delta / (s^T A s)    =", round(2 * delta / (x2 @ F2 @ x2), 6))
print("residual norm after 2    =", f"{np.linalg.norm(g2 - F2 @ x2):.1e}")
print("g . s (equals s^T A s)   =", round(g2 @ x2, 6))

# ---- 2. Softmax policy: Fisher matrix, natural gradient step, line search ----
rng = np.random.default_rng(0)
N, K, d = 6, 3, 3
phi = rng.normal(size=(N, K, d))
adv = rng.normal(size=(N, K))
theta_old = rng.normal(size=d) * 2.0


def probs(theta):
    z = phi @ theta
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def surrogate(theta):
    """L(theta) - eta, up to the constant 1/(1-gamma) factor; the objective of Equation (14) with A for Q."""
    return np.mean(np.sum(probs(theta) * adv, axis=1))


def mean_kl(theta_a, theta_b):
    pa, pb = probs(theta_a), probs(theta_b)
    return np.mean(np.sum(pa * (np.log(pa) - np.log(pb)), axis=1))


def gradient(theta):
    p = probs(theta)
    # d/dtheta sum_a p_a A_a = sum_a p_a A_a (phi_a - phi_bar)
    phi_bar = np.einsum("nk,nkd->nd", p, phi)
    return np.einsum("nk,nk,nkd->d", p, adv, phi - phi_bar[:, None, :]) / N


def fisher(theta):
    p = probs(theta)
    phi_bar = np.einsum("nk,nkd->nd", p, phi)
    c = phi - phi_bar[:, None, :]
    return np.einsum("nk,nki,nkj->ij", p, c, c) / N


F = fisher(theta_old)
g = gradient(theta_old)

# Finite-difference Hessian of the mean KL at theta_old (second derivative, Appendix C)
h = 1e-4
H = np.zeros((d, d))
for i in range(d):
    for j in range(d):
        ei, ej = np.eye(d)[i] * h, np.eye(d)[j] * h
        H[i, j] = (
            mean_kl(theta_old, theta_old + ei + ej)
            - mean_kl(theta_old, theta_old + ei - ej)
            - mean_kl(theta_old, theta_old - ei + ej)
            + mean_kl(theta_old, theta_old - ei - ej)
        ) / (4 * h * h)

print()
print("== Softmax policy, N=6 states, K=3 actions, d=3 parameters")
print("g (gradient of L)        =", g)
print("Fisher matrix F          =", F.tolist())
print("max |F - Hessian of KL|  =", f"{np.abs(F - H).max():.1e}")
print("eigenvalues of F         =", np.linalg.eigvalsh(F))

s = conjugate_gradient(lambda v: F @ v, g, 10)
print("search direction s = F^-1 g (CG, 10 iters) =", s)
print("np.linalg.solve(F, g)    =", np.linalg.solve(F, g))

for delta in (0.01, 0.1):
    beta = np.sqrt(2 * delta / (s @ F @ s))
    theta_nat = theta_old + beta * s
    # Plain gradient step with the same quadratic-KL budget
    beta_g = np.sqrt(2 * delta / (g @ F @ g))
    theta_grad = theta_old + beta_g * g
    print(f"\ndelta = {delta}")
    print("  beta                   =", round(beta, 4))
    print("  natural step: actual KL =", round(mean_kl(theta_old, theta_nat), 5), " L gain =", round(surrogate(theta_nat) - surrogate(theta_old), 5))
    print("  gradient step (same predicted KL): actual KL =", round(mean_kl(theta_old, theta_grad), 5), " L gain =", round(surrogate(theta_grad) - surrogate(theta_old), 5))
    print("  first-order gain g.(beta s) =", round(g @ (beta * s), 5))

# Line search: shrink beta exponentially until the KL constraint holds and L improves
print("\n== Line search (shrink factor 0.5) from the full step, delta = 0.1")
delta = 0.1
beta = np.sqrt(2 * delta / (s @ F @ s))
L0 = surrogate(theta_old)
for k in range(10):
    th = theta_old + (0.5**k) * beta * s
    kl_k, L_k = mean_kl(theta_old, th), surrogate(th)
    ok = kl_k <= delta and L_k > L0
    print(f"  k={k}  step scale {0.5**k:.4f}  KL {kl_k:.4f}  L gain {L_k - L0:+.4f}  accept {ok}")
    if ok:
        break
