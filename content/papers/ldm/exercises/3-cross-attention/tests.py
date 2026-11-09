import numpy as np
from submission import cross_attention

PHI = np.array([[1.0, 0.0], [0.0, 1.0]])
TAU = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
EYE = np.eye(2)
WV = np.array([[1.0, 0.0], [0.0, 2.0]])


def reference(phi, tau, Wq, Wk, Wv):
    d = Wq.shape[0]
    out = np.zeros((phi.shape[0], d))
    for i in range(phi.shape[0]):
        q = Wq @ phi[i]
        s = np.array([q @ (Wk @ t) / np.sqrt(d) for t in tau])
        p = np.exp(s - s.max())
        p /= p.sum()
        out[i] = sum(p[j] * (Wv @ tau[j]) for j in range(len(tau)))
    return out


def test_tiny_example_weights():
    _, w = cross_attention(PHI, TAU, EYE, EYE, WV)
    want = np.array([[0.4011, 0.1978, 0.4011], [0.1978, 0.4011, 0.4011]])
    assert np.allclose(w, want, atol=1e-4), f"expected {want}, got {w}"


def test_tiny_example_output():
    out, _ = cross_attention(PHI, TAU, EYE, EYE, WV)
    want = np.array([[0.8022, 1.1978], [0.5989, 1.6044]])
    assert np.allclose(out, want, atol=1e-4), f"expected {want}, got {out}"


def test_shapes_when_n_differs_from_m():
    rng = np.random.default_rng(0)
    phi, tau = rng.normal(size=(5, 3)), rng.normal(size=(4, 2))
    Wq, Wk, Wv = rng.normal(size=(6, 3)), rng.normal(size=(6, 2)), rng.normal(size=(6, 2))
    out, w = cross_attention(phi, tau, Wq, Wk, Wv)
    assert out.shape == (5, 6), f"output expected shape (5, 6), got {out.shape}"
    assert w.shape == (5, 4), f"weights expected shape (5, 4), got {w.shape}"
    assert np.allclose(w.sum(-1), 1), f"each row of weights must sum to 1, got {w.sum(-1)}"


def test_matches_reference_with_distinct_widths():
    rng = np.random.default_rng(1)
    phi, tau = rng.normal(size=(5, 3)), rng.normal(size=(4, 2))
    Wq, Wk, Wv = rng.normal(size=(6, 3)), rng.normal(size=(6, 2)), rng.normal(size=(6, 2))
    out, _ = cross_attention(phi, tau, Wq, Wk, Wv)
    want = reference(phi, tau, Wq, Wk, Wv)
    assert np.allclose(out, want), f"expected {want}, got {out}"


def test_token_order_does_not_matter():
    rng = np.random.default_rng(2)
    phi, tau = rng.normal(size=(4, 3)), rng.normal(size=(6, 2))
    Wq, Wk, Wv = rng.normal(size=(5, 3)), rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
    a, _ = cross_attention(phi, tau, Wq, Wk, Wv)
    b, _ = cross_attention(phi, tau[::-1].copy(), Wq, Wk, Wv)
    assert np.allclose(a, b), f"reversing the tokens must not change the output, got {a} and {b}"


def test_condition_changes_output():
    rng = np.random.default_rng(3)
    phi = rng.normal(size=(4, 3))
    Wq, Wk, Wv = rng.normal(size=(5, 3)), rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
    a, _ = cross_attention(phi, rng.normal(size=(3, 2)), Wq, Wk, Wv)
    b, _ = cross_attention(phi, rng.normal(size=(3, 2)), Wq, Wk, Wv)
    assert not np.allclose(a, b), "two different conditions gave the same output"
