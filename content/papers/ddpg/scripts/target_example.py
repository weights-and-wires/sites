"""Numbers for the "Critic target" example (Algorithm 1, the y_i line).

Run: python3 scripts/target_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

gamma = 0.9
# A minibatch of three transitions with a scalar state and a scalar action.
s = np.array([1.0, 0.0, 2.0])
a = np.array([0.5, 0.5, 1.0])
r = np.array([0.0, 1.0, 2.0])
s_next = np.array([2.0, -1.0, 3.0])
done = np.array([0.0, 0.0, 1.0])


def actor_target(s):  # mu'(s) = 0.5 s
    return 0.5 * s


def critic_target(s, a):  # Q'(s, a) = s - (a - s)^2
    return s - (a - s) ** 2


def critic(s, a):  # the online critic Q(s, a) = 0.2 s + a
    return 0.2 * s + a


a_next = actor_target(s_next)
q_next = critic_target(s_next, a_next)
y = r + gamma * (1 - done) * q_next
y_unmasked = r + gamma * q_next
q = critic(s, a)
err = y - q

print("mu'(s')          =", a_next)
print("Q'(s', mu'(s'))  =", q_next)
print("y (masked)       =", y)
print("y (no mask)      =", y_unmasked)
print("Q(s, a)          =", q)
print("y - Q(s, a)      =", err)
print("squared errors   =", err**2)
print(f"critic loss      = {np.mean(err**2):.5f}")
