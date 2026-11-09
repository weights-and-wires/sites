import numpy as np
from submission import muon_step, newton_schulz


def mat(n, m, seed):
    return np.random.default_rng(seed).normal(size=(n, m))


def sv(M):
    return np.linalg.svd(M, compute_uv=False)


def test_first_step_buffer_is_the_gradient():
    G = mat(4, 6, 0)
    _, B = muon_step(np.zeros((4, 6)), G, np.zeros((4, 6)), lr=0.1, mu=0.9)
    assert np.allclose(B, G), f"with B = 0 the new buffer must equal G; got max difference {np.abs(B - G).max()}"


def test_first_step_update():
    W, G = mat(4, 6, 1), mat(4, 6, 2)
    W_new, _ = muon_step(W, G, np.zeros_like(G), lr=0.1, mu=0.9)
    want = W - 0.1 * newton_schulz(G)
    assert np.allclose(W_new, want, atol=1e-9), f"expected\n{want}\ngot\n{W_new}"


def test_momentum_accumulates_raw_gradients():
    G, B = mat(5, 3, 3), mat(5, 3, 4)
    _, B_new = muon_step(np.zeros((5, 3)), G, B, lr=0.1, mu=0.95)
    want = 0.95 * B + G
    assert np.allclose(B_new, want), f"expected B = mu * B + G =\n{want}\ngot\n{B_new}"


def test_orthogonalize_after_momentum():
    # Orthogonalizing G first and then adding momentum gives a different W.
    W, G, B = mat(4, 4, 5), mat(4, 4, 6), mat(4, 4, 7)
    W_new, _ = muon_step(W, G, B, lr=0.05, mu=0.9)
    want = W - 0.05 * newton_schulz(0.9 * B + G)
    assert np.allclose(W_new, want, atol=1e-9), f"expected NS applied to mu * B + G; max difference {np.abs(W_new - want).max()}"


def test_step_moves_against_the_update():
    W, G = np.zeros((3, 3)), np.eye(3) * 2.0
    W_new, _ = muon_step(W, G, np.zeros((3, 3)), lr=0.1, mu=0.9)
    assert np.trace(W_new) < 0, f"for a positive gradient the weight must decrease; trace of W is {np.trace(W_new)}"


def test_step_has_spectral_scale_lr():
    W, G, B = np.zeros((6, 4)), mat(6, 4, 8), mat(6, 4, 9)
    W_new, _ = muon_step(W, G, B, lr=0.02, mu=0.95)
    s = sv(W - W_new) / 0.02
    assert np.all((s > 0.6) & (s < 1.3)), f"singular values of the step divided by lr must be near 1, got {s}"


def test_gradient_scale_does_not_change_the_step():
    W, G = mat(4, 5, 10), mat(4, 5, 11)
    a, _ = muon_step(W, G, np.zeros_like(G), lr=0.1, mu=0.9)
    b, _ = muon_step(W, 1000.0 * G, np.zeros_like(G), lr=0.1, mu=0.9)
    assert np.allclose(a, b, atol=1e-6), f"Muon ignores the gradient scale at the first step; max difference {np.abs(a - b).max()}"


def test_inputs_are_not_modified():
    W, G, B = mat(3, 4, 12), mat(3, 4, 13), mat(3, 4, 14)
    copies = [x.copy() for x in (W, G, B)]
    muon_step(W, G, B, lr=0.1, mu=0.9)
    for name, x, y in zip("WGB", (W, G, B), copies):
        assert np.array_equal(x, y), f"muon_step changed {name} in place"
