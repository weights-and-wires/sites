"""Numbers for the "combined Rainbow target" example (Section 3).

Run: python3 scripts/rainbow_target.py
"""

import numpy as np

from c51_projection import project

np.set_printoptions(precision=4, suppress=True)

z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
gamma = 0.9
rewards = [1.0, 0.0, 0.0]
n = len(rewards)
R = sum(gamma**k * r for k, r in enumerate(rewards))
gamma_n = gamma**n

# Next-state distributions at S_{t+n}, one row per action (2 actions, 5 atoms).
p_online = np.array([[0.05, 0.15, 0.30, 0.30, 0.20], [0.10, 0.30, 0.40, 0.15, 0.05]])
p_target = np.array([[0.30, 0.30, 0.20, 0.10, 0.10], [0.05, 0.10, 0.20, 0.35, 0.30]])

print("online means  =", p_online @ z)
print("target means  =", p_target @ z)
a_online = int(np.argmax(p_online @ z))
a_target = int(np.argmax(p_target @ z))
print("action chosen by the online network =", a_online)
print("action chosen by the target network =", a_target)
print("R_t^(3) =", R, " gamma^3 =", round(gamma_n, 4))

_, _, m_double = project(p_target[a_online], R, gamma_n, z)
_, _, m_single = project(p_target[a_target], R, gamma_n, z)
print("target m (double: online picks, target scores) =", m_double)
print("target m (target network picks, as in C51)     =", m_single)

# KL priority against a prediction for (S_t, A_t).
p_pred = np.array([0.10, 0.20, 0.30, 0.25, 0.15])
kl = float(np.sum(m_double * (np.log(m_double + 1e-12) - np.log(p_pred))))
ce = float(-np.sum(m_double * np.log(p_pred)))
ent = float(-np.sum(m_double * np.log(m_double + 1e-12)))
print("cross entropy -sum m log p =", round(ce, 4))
print("entropy of m               =", round(ent, 4))
print("KL(m || p)                 =", round(kl, 4))
print("priority with omega = 0.5  =", round(kl**0.5, 4))
