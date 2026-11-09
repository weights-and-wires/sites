import numpy as np
from submission import gated_activation

X = np.array([[1.0, 2.0, 3.0, 4.0]])
WF = np.array([[[1.0]], [[0.5]]])    # shape (2, 1, 1): taps [1.0, 0.5]; 0.5 is on the current sample
WG = np.array([[[0.5]], [[-1.0]]])


def test_page_example():
    # Values from the page (scripts/gated_example.py).
    got = gated_activation(X, WF, WG, 1)
    want = np.array([[0.1243, 0.1759, 0.1190, 0.0759]])
    assert got.shape == (1, 4), f"expected shape (1, 4), got {got.shape}"
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_zero_gate_input_gives_half_of_tanh():
    # With Wg = 0 the gate is sigmoid(0) = 0.5 everywhere.
    Wg = np.zeros_like(WG)
    got = gated_activation(X, WF, Wg, 1)
    want = 0.5 * np.tanh(np.array([[0.5, 2.0, 3.5, 5.0]]))
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_output_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 40)) * 5
    Wf, Wg = rng.normal(size=(2, 4, 3)), rng.normal(size=(2, 4, 3))
    z = gated_activation(x, Wf, Wg, 2)
    assert z.shape == (4, 40), f"expected shape (4, 40), got {z.shape}"
    assert z.min() > -1 and z.max() < 1, f"tanh * sigmoid lies in (-1, 1); got [{z.min()}, {z.max()}]"


def test_sign_follows_filter():
    # sigmoid is positive, so z has the sign of the tanh branch.
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 30))
    Wf, Wg = rng.normal(size=(2, 3, 2)), rng.normal(size=(2, 3, 2))
    z = gated_activation(x, Wf, Wg, 1)
    f = np.zeros((3, 30))
    for j in range(2):
        shifted = np.zeros_like(x)
        shift = 1 - j
        shifted[:, shift:] = x[:, : 30 - shift]
        f += Wf[j] @ shifted
    assert np.array_equal(np.sign(z), np.sign(f)), "the sign of z must equal the sign of the filter convolution"


def test_filter_and_gate_use_different_weights():
    # Swapping Wf and Wg must change the answer.
    a = gated_activation(X, WF, WG, 1)
    b = gated_activation(X, WG, WF, 1)
    assert not np.allclose(a, b), "swapping filter and gate weights should change the output"


def test_causal_with_dilation():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 25))
    Wf, Wg = rng.normal(size=(2, 3, 2)), rng.normal(size=(2, 3, 2))
    base = gated_activation(x, Wf, Wg, 4)
    x2 = x.copy()
    x2[:, 15:] += 3.0
    new = gated_activation(x2, Wf, Wg, 4)
    assert np.allclose(base[:, :15], new[:, :15]), "outputs before t=15 must not depend on x[:, 15:]"
    assert not np.allclose(base[:, 15:], new[:, 15:]), "outputs from t=15 on should change"


def test_dilation_applies_to_both_branches():
    # Reference computed sample by sample from Equation 2 with dilation 2.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 12))
    Wf, Wg = rng.normal(size=(2, 3, 2)), rng.normal(size=(2, 3, 2))
    d = 2
    want = np.zeros((3, 12))
    for t in range(12):
        f = np.zeros(3)
        g = np.zeros(3)
        for j in range(2):
            s = t - (1 - j) * d
            if s >= 0:
                f += Wf[j] @ x[:, s]
                g += Wg[j] @ x[:, s]
        want[:, t] = np.tanh(f) * (1 / (1 + np.exp(-g)))
    got = gated_activation(x, Wf, Wg, d)
    assert np.allclose(got, want), f"max difference from the sample-by-sample reference: {np.abs(got - want).max()}"
