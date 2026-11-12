"""Numbers for the "Selection with a prior" example.

Run: python3 scripts/puct_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

C_PUCT = 5.0  # Extended Data Table 5


def puct_scores(Q, N, P, c_puct):
    """Q + u with u = c * P * sqrt(sum_b N(s, b)) / (1 + N(s, a)) (Methods, "Search algorithm")."""
    u = c_puct * P * np.sqrt(N.sum()) / (1.0 + N)
    return Q + u, u


# One node with three moves.
Q = np.array([0.2, 0.5, 0.0])
N = np.array([10.0, 2.0, 0.0])
P = np.array([0.6, 0.3, 0.1])
score, u = puct_scores(Q, N, P, C_PUCT)
print("sqrt(sum N)      =", round(float(np.sqrt(N.sum())), 4))
print("bonus u          =", u)
print("Q + u            =", score)
print("selected action  =", int(np.argmax(score)))

# A fresh node: no visits yet, so sqrt(0) = 0 and the bonus vanishes.
fresh_score, fresh_u = puct_scores(np.zeros(3), np.zeros(3), P, C_PUCT)
print("fresh node bonus =", fresh_u, "-> argmax picks action", int(np.argmax(fresh_score)))

# Long run on a bandit-like node: the true values are fixed, and the prior
# favors the worst move. Selection starts with the prior and ends with the values.
true_q = np.array([0.1, 0.3, 0.5])
P2 = np.array([0.7, 0.2, 0.1])
N2 = np.zeros(3)
W2 = np.zeros(3)
for t in range(1, 1001):
    Q2 = np.divide(W2, N2, out=np.zeros(3), where=N2 > 0)
    # Start with one visit's worth of sqrt so that the first pick follows the prior.
    s = Q2 + C_PUCT * P2 * np.sqrt(max(N2.sum(), 1.0)) / (1.0 + N2)
    a = int(np.argmax(s))
    N2[a] += 1
    W2[a] += true_q[a]
    if t in (10, 100, 1000):
        print(f"after {t:4d} simulations, visit share =", N2 / N2.sum())
