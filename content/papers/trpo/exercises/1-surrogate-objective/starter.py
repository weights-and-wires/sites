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
    raise NotImplementedError


def advantages(P, r, pi, gamma):
    """A_pi(s, a) = Q_pi(s, a) - V_pi(s), as an (S, A) array. Here Q = r(s) + gamma * sum_s' P V(s')."""
    raise NotImplementedError


def surrogate(P, r, rho0, gamma, pi, pi_new):
    """L_pi(pi_new) = eta(pi) + sum_s rho_pi(s) sum_a pi_new(a|s) A_pi(s, a) (Equation 3)."""
    raise NotImplementedError
