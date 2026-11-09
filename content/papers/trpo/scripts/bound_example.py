"""Compute the numbers for the surrogate objective and monotonic-improvement example.

A two-state MDP with exact policy evaluation. The notation follows the paper
(Section 2): reward r(s), discount gamma, start distribution rho_0, unnormalized
discounted visitation rho_pi, and advantage A_pi.

Run: python3 scripts/bound_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

gamma = 0.9
r = np.array([0.0, 1.0])  # r(s): state 1 pays 1
rho0 = np.array([1.0, 0.0])  # start in state 0
# P[s, a, s']: action 0 stays, action 1 switches to the other state
P = np.zeros((2, 2, 2))
P[0, 0, 0] = P[1, 0, 1] = 1.0
P[0, 1, 1] = P[1, 1, 0] = 1.0


def policy(p_switch):
    """pi[s, a] from the probability of action 1 (switch) in each state."""
    p = np.asarray(p_switch, dtype=float)
    return np.stack([1 - p, p], axis=1)


def evaluate(pi):
    """Return eta, rho (unnormalized visitation), V, Q, A for policy pi."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    G = np.linalg.inv(np.eye(2) - gamma * P_pi)  # sum_t (gamma P_pi)^t
    V = G @ r
    rho = G.T @ rho0  # Equation (2): sum_t gamma^t P(s_t = s)
    Q = r[:, None] + gamma * np.einsum("sat,t->sa", P, V)
    A = Q - V[:, None]
    return rho0 @ V, rho, V, Q, A


def surrogate(pi, pi_new):
    """L_pi(pi_new) from Equation (3)."""
    eta, rho, _, _, A = evaluate(pi)
    return eta + np.sum(rho[:, None] * pi_new * A)


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def kl(p, q):
    return np.sum(p * np.log(p / q), axis=-1)


pi = policy([0.5, 0.2])
pi_new = policy([0.9, 0.2])

eta, rho, V, Q, A = evaluate(pi)
eta_new, rho_new, _, _, _ = evaluate(pi_new)
print("pi      (P[switch] per state) =", pi[:, 1])
print("pi_new  (P[switch] per state) =", pi_new[:, 1])
print("eta(pi)                       =", round(eta, 4))
print("eta(pi_new)                   =", round(eta_new, 4))
print("rho_pi (sums to 1/(1-gamma))  =", rho, "sum", round(rho.sum(), 4))
print("V_pi                          =", V)
print("A_pi[s, a]                    =", A.tolist())

# Equation (2): eta(pi_new) = eta(pi) + sum_s rho_pi_new(s) sum_a pi_new(a|s) A_pi(s,a)
eq2 = eta + np.sum(rho_new[:, None] * pi_new * A)
print("Equation (2) right-hand side  =", round(eq2, 4))

L = surrogate(pi, pi_new)
print("L_pi(pi_new), Equation (3)    =", round(L, 4))
print("L_pi(pi)                      =", round(surrogate(pi, pi), 4))

eps = np.abs(A).max()
C = 4 * eps * gamma / (1 - gamma) ** 2
d_tv = tv(pi, pi_new).max()
d_kl = kl(pi, pi_new).max()
print()
print("epsilon = max |A|             =", round(eps, 4))
print("C = 4 eps gamma/(1-gamma)^2   =", round(C, 4))
print("D_TV^max (Equation 7)         =", round(d_tv, 4))
print("D_KL^max                      =", round(d_kl, 4))
print("D_TV^2 <= D_KL                =", d_tv**2 <= d_kl)
print("KL(pi || pi_new) in state 0   =", round(kl(pi, pi_new)[0], 4))
print("KL(pi_new || pi) in state 0   =", round(kl(pi_new, pi)[0], 4), "(not symmetric)")
b8 = L - 4 * eps * gamma / (1 - gamma) ** 2 * d_tv**2
b9 = L - C * d_kl
print("Theorem 1 bound (Equation 8)  =", round(b8, 4))
print("KL bound (Equation 9)         =", round(b9, 4))
print("true improvement eta_new-eta  =", round(eta_new - eta, 4))
print("surrogate gain L - eta        =", round(L - eta, 4))

print()
print("Sweep: move P[switch | s=0] from 0.5 toward 1 (the penalized objective M_i).")
print("p_new   eta_new   L-eta     D_KL^max  M-eta     eta_new-eta")
best = (None, -1e9)
for p in [0.5, 0.52, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99]:
    new = policy([p, 0.2])
    e_new = evaluate(new)[0]
    Lp = surrogate(pi, new)
    dk = kl(pi, new).max()
    M = Lp - C * dk
    print(f"{p:.2f}   {e_new:.4f}   {Lp - eta:+.4f}   {dk:.4f}    {M - eta:+.4f}   {e_new - eta:+.4f}")
    assert e_new >= M - 1e-12

print()
print("Penalty with the theoretical C versus a hard constraint D_KL^max <= delta.")
grid = np.linspace(0.5, 0.999, 4991)
M_vals = np.array([surrogate(pi, policy([p, 0.2])) - C * kl(pi, policy([p, 0.2])).max() for p in grid])
p_pen = grid[M_vals.argmax()]
print(f"penalty argmax p (C = {C:.1f})      =", round(p_pen, 4), " eta_new - eta =", round(evaluate(policy([p_pen, 0.2]))[0] - eta, 4))
delta = 0.01
ok = [p for p in grid if kl(pi, policy([p, 0.2])).max() <= delta]
p_con = max(ok)
print(f"constraint argmax p (delta = {delta})  =", round(p_con, 4), " eta_new - eta =", round(evaluate(policy([p_con, 0.2]))[0] - eta, 4))
