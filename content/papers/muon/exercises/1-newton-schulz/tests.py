import numpy as np
from submission import newton_schulz

TUNED = (3.4445, -4.7750, 2.0315)
BASELINE = (2.0, -1.5, 0.5)


def phi(x, coeffs):
    a, b, c = coeffs
    return a * x + b * x**3 + c * x**5


def make(n, m, seed=0):
    return np.random.default_rng(seed).normal(size=(n, m))


def sv(M):
    return np.linalg.svd(M, compute_uv=False)


def test_shape_for_wide_tall_square():
    for n, m in [(3, 7), (7, 3), (5, 5)]:
        out = newton_schulz(make(n, m))
        assert out.shape == (n, m), f"expected shape {(n, m)}, got {out.shape}"


def test_one_step_maps_singular_values_through_phi():
    # One step is U phi(S / ||G||_F) V^T, so check it against the SVD.
    G = make(4, 6, seed=1)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    want = U @ np.diag(phi(S / np.linalg.norm(G), TUNED)) @ Vt
    got = newton_schulz(G, steps=1)
    assert np.allclose(got, want, atol=1e-5), f"expected\n{want}\ngot\n{got}"


def test_one_step_tall_matrix():
    G = make(6, 4, seed=2)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    want = U @ np.diag(phi(S / np.linalg.norm(G), TUNED)) @ Vt
    got = newton_schulz(G, steps=1)
    assert np.allclose(got, want, atol=1e-5), f"tall input: expected\n{want}\ngot\n{got}"


def test_zero_steps_only_normalizes():
    G = make(3, 5, seed=3)
    got = newton_schulz(G, steps=0)
    want = G / np.linalg.norm(G)
    assert np.allclose(got, want, atol=1e-5), f"expected G / ||G||_F, got\n{got}"


def test_five_steps_put_singular_values_in_the_band():
    G = make(8, 5, seed=4)
    s = sv(newton_schulz(G, steps=5))
    assert np.all((s > 0.6) & (s < 1.3)), f"expected singular values in (0.6, 1.3), got {s}"


def test_baseline_coefficients_converge_to_uv_t():
    G = make(7, 4, seed=5)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    got = newton_schulz(G, steps=40, coeffs=BASELINE)
    assert np.allclose(got, U @ Vt, atol=1e-6), f"expected U V^T\n{U @ Vt}\ngot\n{got}"


def test_scale_invariance():
    G = make(5, 5, seed=6)
    a, b = newton_schulz(G), newton_schulz(1000.0 * G)
    assert np.allclose(a, b, atol=1e-5), f"Ortho(cG) must equal Ortho(G); max difference {np.abs(a - b).max()}"


def test_large_spectral_norm_does_not_blow_up():
    # All entries near 1 means the spectral norm (about 12) far exceeds the largest entry.
    G = np.ones((12, 12)) * 3.0 + make(12, 12, seed=7) * 0.1
    s = sv(newton_schulz(G, steps=5))
    assert s.max() < 1.3, f"largest singular value expected below 1.3, got {s.max()}; normalize by the Frobenius norm"


def test_transpose_equivariance():
    G = make(3, 9, seed=8)
    a, b = newton_schulz(G), newton_schulz(G.T)
    assert np.allclose(a.T, b, atol=1e-9), f"NS(G^T) must equal NS(G)^T; max difference {np.abs(a.T - b).max()}"


def test_input_is_not_modified():
    G = make(4, 4, seed=9)
    before = G.copy()
    newton_schulz(G)
    assert np.array_equal(G, before), "newton_schulz changed its input in place"
