import numpy as np
from submission import conformer_block


def ident(v):
    return v


def test_hand_worked_scalar():
    # x = 1. ffn1: 1 + 0.5 * 2 = 2. mhsa(x) = x^2: 2 + 4 = 6.
    # conv(x) = -x / 2: 6 - 3 = 3. ffn2(x) = 4x: 3 + 0.5 * 12 = 9.
    x = np.array([[1.0]])
    got = conformer_block(x, lambda v: 2 * v, lambda v: v ** 2, lambda v: -v / 2, lambda v: 4 * v, ident)
    assert np.allclose(got, [[9.0]]), f"expected [[9.0]], got {got}"


def test_modules_run_in_order():
    calls = []

    def rec(name):
        def f(v):
            calls.append(name)
            return 0 * v
        return f

    conformer_block(np.ones((2, 2)), rec("ffn1"), rec("mhsa"), rec("conv"), rec("ffn2"), ident)
    want = ["ffn1", "mhsa", "conv", "ffn2"]
    assert calls == want, f"expected call order {want}, got {calls}"


def test_zero_modules_give_identity():
    x = np.arange(6.0).reshape(3, 2)
    zero = lambda v: np.zeros_like(v)
    got = conformer_block(x, zero, zero, zero, zero, ident)
    assert np.allclose(got, x), f"expected the input back, got {got}"


def test_only_the_ffn_residuals_are_halved():
    x = np.ones((1, 1))
    one = lambda v: np.ones_like(v)
    zero = lambda v: np.zeros_like(v)
    assert np.allclose(conformer_block(x, one, zero, zero, zero, ident), 1.5), "expected 1.5 from ffn1 alone"
    assert np.allclose(conformer_block(x, zero, one, zero, zero, ident), 2.0), "expected 2.0 from mhsa alone"
    assert np.allclose(conformer_block(x, zero, zero, one, zero, ident), 2.0), "expected 2.0 from conv alone"
    assert np.allclose(conformer_block(x, zero, zero, zero, one, ident), 1.5), "expected 1.5 from ffn2 alone"


def test_final_layer_norm_is_applied_last():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])

    def ln(v):
        m = v.mean(axis=1, keepdims=True)
        s = v.std(axis=1, keepdims=True)
        return (v - m) / s

    zero = lambda v: np.zeros_like(v)
    got = conformer_block(x, zero, zero, zero, zero, ln)
    want = ln(x)
    assert np.allclose(got, want), f"expected {want}, got {got}"
    assert abs(got.mean()) < 1e-9, f"expected zero mean after the layer norm, got {got.mean()}"
