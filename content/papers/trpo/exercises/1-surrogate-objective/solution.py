import numpy as np


def policy_value(P, r, pi, gamma):
    """Finished for you. Return (P_pi, V_pi).

    P has shape (S, A, S) with P[s, a, s'] = P(s' | s, a). r has shape (S,).
    pi has shape (S, A). P_pi[s, s'] = sum_a pi[s, a] P[s, a, s'], and V_pi solves
    V = r + gamma * P_pi V.
    """
    P_pi = np.einsum("sa,sat->st", pi, P)
    V = np.linalg.solve(np.eye(len(r)) - gamma * P_pi, r)
    return P_pi, V


def visitation(P, rho0, pi, gamma):
    """Unnormalized discounted state visitation rho_pi(s) = sum_t gamma^t P(s_t = s) (Section 2)."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    # rho^T = rho0^T (I - gamma P_pi)^-1, so solve with the transpose.
    return np.linalg.solve((np.eye(len(rho0)) - gamma * P_pi).T, rho0)


def advantages(P, r, pi, gamma):
    """A_pi(s, a) = Q_pi(s, a) - V_pi(s), as an (S, A) array. Here Q = r(s) + gamma * sum_s' P V(s')."""
    _, V = policy_value(P, r, pi, gamma)
    Q = r[:, None] + gamma * np.einsum("sat,t->sa", P, V)
    return Q - V[:, None]


def surrogate(P, r, rho0, gamma, pi, pi_new):
    """L_pi(pi_new) = eta(pi) + sum_s rho_pi(s) sum_a pi_new(a|s) A_pi(s, a) (Equation 3)."""
    _, V = policy_value(P, r, pi, gamma)
    eta = rho0 @ V
    rho = visitation(P, rho0, pi, gamma)
    A = advantages(P, r, pi, gamma)
    return eta + np.sum(rho[:, None] * pi_new * A)
