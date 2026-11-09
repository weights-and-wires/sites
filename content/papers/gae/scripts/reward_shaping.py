"""Reward shaping and GAE (Section 4 of the paper).

Part 1 checks Equations 22 to 24 on an exact, small MDP: shaping the reward with a
potential Phi subtracts Phi(s) from Q and V and leaves the advantage unchanged.

Part 2 checks Equation 25 on the trajectory from the page: the shaped reward with
Phi = V equals the TD residual, so the (gamma lambda)-discounted sum of shaped
rewards is GAE.

Run: python3 scripts/reward_shaping.py
"""

import numpy as np

np.set_printoptions(precision=5, suppress=True)

# Part 1: Equations 20 to 24 on a random MDP with reward r(s, a).
S, A_N, gamma = 4, 3, 0.9
rng = np.random.default_rng(0)
P = rng.dirichlet(np.ones(S), size=(S, A_N))  # P[s, a, s']
r = rng.normal(size=(S, A_N))  # r(s, a)
pi = rng.dirichlet(np.ones(A_N), size=S)
Phi = rng.normal(size=S)


def solve(reward):
    """Exact V, Q, A of pi for reward[s, a]."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.sum(pi * reward, axis=1)
    V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
    Q = reward + gamma * np.einsum("sat,t->sa", P, V)
    return V, Q, Q - V[:, None]


# r_tilde(s, a) = E_{s'}[ r(s, a) + gamma Phi(s') - Phi(s) ]   (Equation 20, averaged over s')
r_tilde = r + gamma * np.einsum("sat,t->sa", P, Phi) - Phi[:, None]
V, Q, A = solve(r)
Vt, Qt, At = solve(r_tilde)
print("Equation 22: max |Q_tilde - (Q - Phi)| =", np.abs(Qt - (Q - Phi[:, None])).max())
print("Equation 23: max |V_tilde - (V - Phi)| =", np.abs(Vt - (V - Phi)).max())
print("Equation 24: max |A_tilde - A|         =", np.abs(At - A).max())
r_v = r + gamma * np.einsum("sat,t->sa", P, V) - V[:, None]
print("Phi = V: V_tilde =", np.round(solve(r_v)[0], 10) + 0.0, "(zero in every state)")

# Part 2: Equation 25 on the page trajectory.
gamma = 0.9
rew = np.array([1.0, 0.0, 0.0, 2.0])
Vs = np.array([0.5, 0.4, 0.6, 0.3])
T = len(rew)
shaped = rew + gamma * np.append(Vs[1:], 0.0) - Vs  # Equation 20 with Phi = V, Phi(terminal) = 0
print()
print("shaped rewards r_tilde_t =", shaped)
for lam in (0.0, 0.5, 1.0):
    total = [sum((gamma * lam) ** l * shaped[t + l] for l in range(T - t)) for t in range(T)]
    print(f"discounted sum of shaped rewards, gamma*lambda = {gamma * lam:.2f}:", np.array(total))
