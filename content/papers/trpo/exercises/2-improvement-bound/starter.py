import numpy as np


def policy_value(P, r, pi, gamma):
    """Finished for you. Return (P_pi, V_pi) for P[s, a, s'], r[s], and pi[s, a]."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    V = np.linalg.solve(np.eye(len(r)) - gamma * P_pi, r)
    return P_pi, V


def surrogate_and_epsilon(P, r, rho0, gamma, pi, pi_new):
    """Finished for you. Return (eta(pi), L_pi(pi_new), epsilon).

    L_pi(pi_new) is Equation 3. epsilon = max_{s,a} |A_pi(s, a)| is the constant in Theorem 1.
    """
    P_pi, V = policy_value(P, r, pi, gamma)
    eta = rho0 @ V
    rho = np.linalg.solve((np.eye(len(r)) - gamma * P_pi).T, rho0)
    Q = r[:, None] + gamma * np.einsum("sat,t->sa", P, V)
    A = Q - V[:, None]
    L = eta + np.sum(rho[:, None] * pi_new * A)
    return eta, L, np.abs(A).max()


def max_tv(pi, pi_new):
    """D_TV^max(pi, pi_new) = max_s 0.5 * sum_a |pi(a|s) - pi_new(a|s)| (Equation 7)."""
    raise NotImplementedError


def max_kl(pi, pi_new):
    """D_KL^max(pi, pi_new) = max_s KL(pi(.|s) || pi_new(.|s)). The old policy comes first."""
    raise NotImplementedError


def penalty_constant(epsilon, gamma):
    """C = 4 * epsilon * gamma / (1 - gamma)^2 (Equation 9)."""
    raise NotImplementedError


def improvement_bound(P, r, rho0, gamma, pi, pi_new, divergence="kl"):
    """Lower bound on eta(pi_new).

    With divergence="tv", return L - C * D_TV^max(pi, pi_new)^2 (Theorem 1, Equation 8).
    With divergence="kl", return L - C * D_KL^max(pi, pi_new) (Equation 9).
    """
    raise NotImplementedError
