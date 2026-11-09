import numpy as np
from submission import ffn_hidden, silu, swiglu_ffn

X = np.array([1.0, 2.0])
W1 = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
V = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
W2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_silu_values():
    got = silu(np.array([-10.0, 0.0, 1.0, 10.0]))
    want = np.array([-4.54e-4, 0.0, 0.7311, 9.9995])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_silu_is_not_relu_or_sigmoid():
    got = silu(np.array([-1.0]))
    assert np.allclose(got, -0.2689, atol=1e-4), f"expected -0.2689 (SiLU dips below zero), got {got}"


def test_hand_example():
    got = swiglu_ffn(X, W1, V, W2)
    want = np.array([2.1932, 6.7469])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_gate_is_on_the_first_branch():
    got = swiglu_ffn(X, W1, V, W2)
    wrong = np.array([2.4927, 7.4770])
    assert not np.allclose(got, wrong, atol=1e-3), f"SiLU is on the value branch: got {got}; it belongs on x W1"


def test_batch_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, 2))
    y = swiglu_ffn(x, W1, V, W2)
    assert y.shape == (2, 5, 2), f"expected shape (2, 5, 2), got {y.shape}"
    assert np.allclose(y[1, 3], swiglu_ffn(x[1, 3], W1, V, W2)), "each position must be processed on its own"


def test_hidden_widths_behind_table_2():
    got = [ffn_hidden(d) for d in (4096, 5120, 6656, 8192)]
    want = [11008, 13824, 17920, 22016]
    assert got == want, f"expected {want}, got {got}"


def test_hidden_is_two_thirds_of_4d():
    assert ffn_hidden(4096, multiple_of=1) == 10922, f"expected 10922 = int(2 * 4 * 4096 / 3), got {ffn_hidden(4096, multiple_of=1)}"
    d = 4096
    ratio = 3 * d * ffn_hidden(d, 1) / (2 * d * 4 * d)
    assert abs(ratio - 1) < 1e-3, f"three matrices of width h must cost about the same as two of width 4d; ratio {ratio:.4f}"
    assert ffn_hidden(100, multiple_of=8) % 8 == 0, f"expected a multiple of 8, got {ffn_hidden(100, multiple_of=8)}"
