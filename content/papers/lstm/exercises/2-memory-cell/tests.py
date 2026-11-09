import math

import numpy as np
from submission import f, g, h, memory_cell_step, run_cell


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def scalar_params(w_c=(1.0, 0.0), b_c=0.0, w_in=(0.0, 0.0), b_in=2.0, w_out=(0.0, 0.0), b_out=-1.0):
    return {
        "W_c": np.array([w_c], dtype=float), "b_c": np.array([b_c]),
        "W_in": np.array([w_in], dtype=float), "b_in": np.array([b_in]),
        "W_out": np.array([w_out], dtype=float), "b_out": np.array([b_out]),
    }


def reference(xs, p):
    """Plain-Python scalar reference of Equations A.5 to A.7."""
    y, s, ys, ss = 0.0, 0.0, [], []
    for x in xs:
        z = (float(x[0]), y)
        i = sig(p["W_in"][0] @ z + p["b_in"][0])
        o = sig(p["W_out"][0] @ z + p["b_out"][0])
        gv = 4 * sig(p["W_c"][0] @ z + p["b_c"][0]) - 2
        s = s + i * gv
        y = o * (2 * sig(s) - 1)
        ys.append(y)
        ss.append(s)
    return np.array(ys), np.array(ss)


def test_ranges_of_f_g_h():
    big = np.array([-50.0, 50.0])
    assert np.allclose(f(big), [0, 1]), f"f must range over (0, 1); expected [0, 1], got {f(big)}"
    assert np.allclose(g(big), [-2, 2]), f"g must range over (-2, 2); expected [-2, 2], got {g(big)}"
    assert np.allclose(h(big), [-1, 1]), f"h must range over (-1, 1); expected [-1, 1], got {h(big)}"


def test_first_step_values():
    p = scalar_params()
    y, s = memory_cell_step(np.array([1.0, 0.0]), np.zeros(1), p)
    assert np.allclose(s, [0.8141], atol=1e-4), f"expected s = 0.8141, got {s}"
    assert np.allclose(y, [0.1038], atol=1e-4), f"expected y = 0.1038, got {y}"


def test_state_is_purely_additive():
    p = scalar_params()
    z = np.array([1.0, 0.0])
    _, s0 = memory_cell_step(z, np.zeros(1), p)
    _, s5 = memory_cell_step(z, np.array([5.0]), p)
    assert np.allclose(s5 - 5.0, s0), f"the state must add to s_prev with weight 1; expected increment {s0}, got {s5 - 5.0}"


def test_closed_input_gate_keeps_state():
    p = scalar_params(b_in=-30.0, w_c=(5.0, 0.0))
    _, s = memory_cell_step(np.array([3.0, 0.0]), np.array([0.7]), p)
    assert np.allclose(s, [0.7], atol=1e-9), f"a closed input gate must leave s at 0.7, got {s}"


def test_zero_cell_input_keeps_state():
    p = scalar_params()
    _, s = memory_cell_step(np.array([0.0, 0.0]), np.array([1.25]), p)
    assert np.allclose(s, [1.25]), f"g(0) = 0 so s must stay at 1.25, got {s}"


def test_closed_output_gate_hides_state():
    p = scalar_params(b_out=-30.0)
    y, s = memory_cell_step(np.array([1.0, 0.0]), np.array([2.0]), p)
    assert np.allclose(y, [0.0], atol=1e-9), f"a closed output gate must give y = 0, got {y}"
    assert s[0] > 2.5, f"the state must still be stored; expected above 2.5, got {s}"


def test_output_uses_updated_state():
    p = scalar_params(b_out=30.0)
    y, s = memory_cell_step(np.array([1.0, 0.0]), np.array([0.0]), p)
    want = 2 * sig(s[0]) - 1
    assert np.allclose(y, [want]), f"with the output gate open, y = h(s_new) = {want}, got {y}"


def test_run_cell_trace_matches_page():
    xs = np.array([[1.0], [2.0], [-1.0], [0.0], [0.0]])
    ys, ss = run_cell(xs, scalar_params())
    assert ys.shape == (5, 1) and ss.shape == (5, 1), f"expected shapes (5, 1), got {ys.shape} and {ss.shape}"
    assert np.allclose(ss[:, 0], [0.8141, 2.1557, 1.3416, 1.3416, 1.3416], atol=1e-4), f"expected s = [0.8141, 2.1557, 1.3416, 1.3416, 1.3416], got {ss[:, 0]}"
    assert np.allclose(ys[:, 0], [0.1038, 0.2131, 0.1575, 0.1575, 0.1575], atol=1e-4), f"expected y = [0.1038, 0.2131, 0.1575, 0.1575, 0.1575], got {ys[:, 0]}"


def test_recurrent_connection_is_used():
    p = scalar_params(w_c=(0.5, 2.0), w_in=(0.0, 1.0), w_out=(0.0, -1.0), b_out=1.0)
    xs = np.array([[1.0], [0.5], [-0.3], [0.8]])
    ys, ss = run_cell(xs, p)
    ry, rs = reference(xs, p)
    assert np.allclose(ys[:, 0], ry), f"expected y = {ry}, got {ys[:, 0]}"
    assert np.allclose(ss[:, 0], rs), f"expected s = {rs}, got {ss[:, 0]}"


def test_multiple_cells_are_independent():
    rng = np.random.default_rng(0)
    n, d_x = 3, 2
    p = {k: rng.normal(size=(n, d_x + n)) for k in ("W_c", "W_in", "W_out")}
    p.update({k: rng.normal(size=n) for k in ("b_c", "b_in", "b_out")})
    xs = rng.normal(size=(6, d_x))
    ys, ss = run_cell(xs, p)
    assert ys.shape == (6, n) and ss.shape == (6, n), f"expected shapes (6, {n}), got {ys.shape} and {ss.shape}"
    # Recompute by hand for cell 1 at step 1 with y_0 = 0, s_0 = 0.
    z = np.concatenate([xs[0], np.zeros(n)])
    i = sig(p["W_in"][1] @ z + p["b_in"][1])
    s1 = i * (4 * sig(p["W_c"][1] @ z + p["b_c"][1]) - 2)
    assert np.isclose(ss[0, 1], s1), f"expected s = {s1} for cell 1 at step 1, got {ss[0, 1]}"
