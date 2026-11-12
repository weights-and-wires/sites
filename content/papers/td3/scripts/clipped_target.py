"""Numbers for the "Clipped double Q-learning" example (Section 4.2, Equation 10, Equation 15).

Run: python3 scripts/clipped_target.py
"""

import numpy as np

gamma = 0.9
r = np.array([1.0, 0.0, 2.0])
done = np.array([0.0, 0.0, 1.0])
q1 = np.array([2.0, 5.0, -1.0])   # Q'_1(s', a~)
q2 = np.array([3.0, 4.0, -2.0])   # Q'_2(s', a~)

y_clip = r + gamma * (1 - done) * np.minimum(q1, q2)
y_mean = r + gamma * (1 - done) * (q1 + q2) / 2
y_one = r + gamma * (1 - done) * q1
print("min(Q1', Q2'):", np.minimum(q1, q2))
print("clipped target y:", y_clip)
print("target with Q1' alone:", y_one)
print("target with the mean:", y_mean)

q1_now = np.array([2.5, 3.0, 1.5])
q2_now = np.array([2.0, 3.5, 2.5])
print("critic 1 loss:", round(float(np.mean((y_clip - q1_now) ** 2)), 5))
print("critic 2 loss:", round(float(np.mean((y_clip - q2_now) ** 2)), 5))
print("sum of the two losses:", round(float(np.mean((y_clip - q1_now) ** 2) + np.mean((y_clip - q2_now) ** 2)), 5))
