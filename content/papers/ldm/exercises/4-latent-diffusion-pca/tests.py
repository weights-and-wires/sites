import numpy as np
from submission import fit_pca, encode, decode, analytic_eps, generate

rng = np.random.default_rng(0)
D, K, N = 16, 2, 4000
BASIS, _ = np.linalg.qr(rng.standard_normal((D, K)))
X = (rng.standard_normal((N, K)) * np.array([3.0, 1.0])) @ BASIS.T + 0.05 * rng.standard_normal((N, D)) + 2.0
BETAS = np.linspace(1e-3, 0.05, 200)


def test_components_are_orthonormal():
    mean, W = fit_pca(X, K)
    assert W.shape == (K, D), f"expected W of shape {(K, D)}, got {W.shape}"
    assert np.allclose(W @ W.T, np.eye(K), atol=1e-8), f"rows of W must be orthonormal, got {W @ W.T}"


def test_reconstruction_error_is_discarded_variance():
    mean, W = fit_pca(X, K)
    xhat = decode(encode(X, mean, W), mean, W)
    err = ((X - xhat) ** 2).sum(1).mean()
    eig = np.sort(np.linalg.eigvalsh(np.cov(X - X.mean(0), rowvar=False, bias=True)))[::-1]
    want = eig[K:].sum()
    assert abs(err - want) < 1e-6, f"expected error {want:.5f} (sum of discarded eigenvalues), got {err:.5f}"


def test_encode_shape_and_mean():
    mean, W = fit_pca(X, K)
    z = encode(X, mean, W)
    assert z.shape == (N, K), f"expected shape {(N, K)}, got {z.shape}"
    assert np.allclose(z.mean(0), 0, atol=1e-8), f"latents of centered data must have mean 0, got {z.mean(0)}"


def test_decode_adds_the_mean():
    mean, W = fit_pca(X, K)
    got = decode(np.zeros((1, K)), mean, W)
    assert np.allclose(got[0], mean), f"decode(0) must be the data mean, got {got[0]}"


def test_analytic_eps_one_dimension():
    got = analytic_eps(np.array([[3.0]]), 0.5, np.array([0.0]), np.array([[4.0]]))
    want = 0.5**0.5 * 3.0 / 2.5
    assert np.allclose(got, [[want]]), f"expected [[{want:.4f}]], got {got}"


def test_analytic_eps_uses_the_mean():
    got = analytic_eps(np.array([[3.0]]), 0.5, np.array([2.0]), np.array([[4.0]]))
    want = 0.5**0.5 * (3.0 - 0.5**0.5 * 2.0) / 2.5
    assert np.allclose(got, [[want]]), f"expected [[{want:.4f}]], got {got}"


def test_analytic_eps_full_covariance():
    Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    m = np.array([1.0, -1.0])
    zt = np.array([[0.3, 0.7], [-1.0, 2.0]])
    ab = 0.36
    St = ab * Sigma + (1 - ab) * np.eye(2)
    want = np.sqrt(1 - ab) * (np.linalg.inv(St) @ (zt - np.sqrt(ab) * m).T).T
    got = analytic_eps(zt, ab, m, Sigma)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_analytic_eps_limits():
    Sigma, m = np.array([[2.0, 0.5], [0.5, 1.0]]), np.array([1.0, -1.0])
    zt = np.array([[0.3, 0.7]])
    assert np.allclose(analytic_eps(zt, 0.0, m, Sigma), zt), "abar = 0 must return z_t"
    assert np.allclose(analytic_eps(zt, 1.0, m, Sigma), 0.0), "abar = 1 must return 0"


def test_generated_samples_match_data_covariance():
    mean, W = fit_pca(X, K)
    Z = encode(X, mean, W)
    m, Sigma = Z.mean(0), np.cov(Z, rowvar=False, bias=True)
    xs = generate(3000, mean, W, m, Sigma, BETAS, seed=1)
    xhat = decode(Z, mean, W)
    cs, cx = np.cov(xs, rowvar=False, bias=True), np.cov(xhat, rowvar=False, bias=True)
    rel = np.linalg.norm(cs - cx) / np.linalg.norm(cx)
    assert rel < 0.15, f"expected relative covariance error below 0.15, got {rel:.3f}"
    assert np.allclose(xs.mean(0), mean, atol=0.25), f"expected sample mean near {mean[:3]}, got {xs.mean(0)[:3]}"
