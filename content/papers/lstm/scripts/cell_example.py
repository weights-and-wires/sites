"""Compute the numbers for the memory-cell example (Section 4.1, Appendix A.1.1).

One memory cell block of size 1 with scalar input. The cell follows
Equations A.5 to A.7 with the functions f, g, h of Equations A.1 to A.3.

Run: python3 scripts/cell_example.py
"""

import numpy as np


def f(x):  # Equation A.1, range (0, 1)
    return 1.0 / (1.0 + np.exp(-x))


def h(x):  # Equation A.2, range (-1, 1)
    return 2.0 / (1.0 + np.exp(-x)) - 1.0


def g(x):  # Equation A.3, range (-2, 2)
    return 4.0 / (1.0 + np.exp(-x)) - 2.0


# Weights act on z = [x_t, y_{t-1}]. All recurrent weights are 0 to keep the trace readable.
w_c, b_c = np.array([1.0, 0.0]), 0.0      # cell input
w_in, b_in = np.array([0.0, 0.0]), 2.0    # input gate
w_out, b_out = np.array([0.0, 0.0]), -1.0  # output gate

xs = [1.0, 2.0, -1.0, 0.0, 0.0]
s, y = 0.0, 0.0
print("range checks: g(10) =", round(float(g(10)), 4), " h(10) =", round(float(h(10)), 4), " f(10) =", round(float(f(10)), 4))
print("t    x     i_in   g(net_c)   s        o_out   h(s)     y")
for t, x in enumerate(xs, start=1):
    z = np.array([x, y])
    i_in = f(w_in @ z + b_in)
    o_out = f(w_out @ z + b_out)
    g_in = g(w_c @ z + b_c)
    s = s + i_in * g_in
    y = o_out * h(s)
    print(f"{t}  {x:5.1f}  {i_in:.4f}  {g_in:8.4f}  {s:8.4f}  {o_out:.4f}  {h(s):7.4f}  {y:7.4f}")

# Closing the input gate: the state is unchanged while i_in is near 0.
for bias in (2.0, -3.0, -6.0):
    print(f"input gate bias {bias:5.1f}: i_in = {f(bias):.5f}; one step adds {f(bias) * g(1.0):.5f} to s for net_c = 1")
