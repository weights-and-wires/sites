"""Compute the numbers for the bidirectional annotation example.

Run: python3 scripts/bidirectional_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# A scalar RNN (n = 1): h_t = tanh(w x_t + u h_{t-1}), h_0 = 0 (Equation 1 with a tanh cell).
# The paper uses gated units (Appendix A.2.1); the plain tanh cell keeps the arithmetic checkable.
w, u = 1.0, 0.5
x = np.array([1.0, 0.0, -1.0])
T = len(x)

fwd = np.zeros(T)
prev = 0.0
for t in range(T):
    prev = np.tanh(w * x[t] + u * prev)
    fwd[t] = prev

bwd_in_reading_order = []  # computed from x_T down to x_1
prev = 0.0
for t in reversed(range(T)):
    prev = np.tanh(w * x[t] + u * prev)
    bwd_in_reading_order.append(prev)
bwd = np.array(bwd_in_reading_order[::-1])  # re-aligned so index j matches x_j

print("forward states:", fwd)
print("backward states as computed (x_3 first):", np.array(bwd_in_reading_order))
print("backward states re-aligned to positions:", bwd)
print("annotations h_j = [fwd_j, bwd_j]:")
print(np.stack([fwd, bwd], axis=1))
