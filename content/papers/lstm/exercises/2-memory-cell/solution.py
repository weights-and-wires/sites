import numpy as np


def f(x):
    """Logistic sigmoid for the gate units (Equation A.1), range (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def g(x):
    """Squashing function for the cell input (Equation A.3), range (-2, 2)."""
    return 4.0 / (1.0 + np.exp(-x)) - 2.0


def h(x):
    """Scaling function for the cell output (Equation A.2), range (-1, 1)."""
    return 2.0 / (1.0 + np.exp(-x)) - 1.0


def memory_cell_step(z, s_prev, params):
    """One step of a bank of n memory cell blocks of size 1 (Equations A.5 to A.7).

    z = concatenation of the current input and the previous cell outputs, shape (d,).
    s_prev = previous internal states, shape (n,).
    params holds W_c, W_in, W_out with shape (n, d) and b_c, b_in, b_out with shape (n,).
    Return (y, s): the cell outputs and the new internal states.
    """
    i_gate = f(params["W_in"] @ z + params["b_in"])
    o_gate = f(params["W_out"] @ z + params["b_out"])
    net_c = params["W_c"] @ z + params["b_c"]
    s = s_prev + i_gate * g(net_c)
    y = o_gate * h(s)
    return y, s


def run_cell(xs, params):
    """Run the cells over xs of shape (T, d_x) from y = 0 and s = 0.

    Each step uses z = [x_t, y_{t-1}]. Return (ys, ss), both of shape (T, n).
    """
    n = params["b_c"].shape[0]
    y = np.zeros(n)
    s = np.zeros(n)
    ys, ss = [], []
    for x in xs:
        z = np.concatenate([x, y])
        y, s = memory_cell_step(z, s, params)
        ys.append(y)
        ss.append(s)
    return np.array(ys), np.array(ss)
