"""Numbers for the "Q-network trained on TD targets" example.

Run: python3 scripts/target_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

gamma = 0.9
# A minibatch of three transitions with K = 2 actions. q holds Q(phi_j, .; theta),
# q_next holds Q(phi_{j+1}, .; theta), and the third row of q_next is unused
# because that transition is terminal.
q = np.array([[0.5, 0.2], [0.5, 0.9], [0.1, 0.3]])
a = np.array([1, 0, 0])
r = np.array([0.0, 1.0, -1.0])
q_next = np.array([[0.4, 0.8], [7.0, 7.0], [0.2, 0.1]])
done = np.array([0.0, 1.0, 0.0])

y = r + gamma * (1 - done) * q_next.max(axis=1)
q_taken = q[np.arange(3), a]
err = y - q_taken
print("targets y_j          =", y)
print("Q(phi_j, a_j)        =", q_taken)
print("y_j - Q(phi_j, a_j)  =", err)
print("squared errors       =", err**2)
print(f"mean loss            = {np.mean(err**2):.4f}")
# Gradient of the mean loss with respect to the network output Q(phi_j, a_j),
# with y held fixed. Equation 3 as printed is -1/2 of this, per sample.
print("dL/dQ(phi_j, a_j)    =", -2 * err / 3)
print("Eq. 3 factor (y - Q) =", err)

# The terminal row: the value of q_next must not change y for that row.
y_alt = r + gamma * (1 - done) * np.full(3, 1e6)
print("y_2 (terminal) with q_next = 1e6:", y_alt[1])
