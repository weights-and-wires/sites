import numpy as np
from submission import loss_hat, isoflop_curve


def test_gopher_and_chinchilla():
    g = float(loss_hat(280e9, 300e9))
    c = float(loss_hat(70e9, 1.4e12))
    assert abs(g - 1.9933) < 2e-4, f"Gopher (280B, 300B tokens): expected 1.9933, got {g}"
    assert abs(c - 1.9366) < 2e-4, f"Chinchilla (70B, 1.4T tokens): expected 1.9366, got {c}"


def test_asymmetric_inputs():
    # N and D have different exponents and different coefficients; swapping them changes the answer.
    a = float(loss_hat(1e9, 2e10))
    assert abs(a - 2.5800) < 2e-4, f"(N=1e9, D=2e10): expected 2.5800, got {a}"
    b = float(loss_hat(2e10, 1e9))
    assert abs(b - 3.0581) < 5e-4, f"(N=2e10, D=1e9): expected 3.0581, got {b}"


def test_irreducible_loss_limit():
    v = float(loss_hat(1e30, 1e30))
    assert abs(v - 1.69) < 1e-3, f"as N and D grow the loss tends to E = 1.69, got {v}"


def test_vectorised():
    v = loss_hat(np.array([1e9, 7e10]), np.array([2e10, 1.4e12]))
    assert v.shape == (2,), f"expected shape (2,), got {v.shape}"
    assert np.allclose(v, [2.5800, 1.9366], atol=2e-4), f"expected [2.58, 1.9366], got {v}"


def test_isoflop_tokens():
    d, _ = isoflop_curve(5.88e23, [7e10, 2.8e11])
    assert np.allclose(d, [1.4e12, 3.5e11], rtol=1e-9), f"expected tokens [1.4e12, 3.5e11], got {d}"
    assert np.allclose(6 * np.array([7e10, 2.8e11]) * d, 5.88e23), f"6*N*D must equal the budget, got {6 * np.array([7e10, 2.8e11]) * d}"


def test_isoflop_losses_have_a_valley():
    sizes = np.array([0.4e9, 0.8e9, 1.6e9, 3.2e9, 6.4e9, 12.8e9])
    d, l = isoflop_curve(1e21, sizes)
    want = [2.4024, 2.3500, 2.3294, 2.3384, 2.3762, 2.4433]
    assert np.allclose(l, want, atol=3e-4), f"expected {want}, got {np.round(l, 4)}"
    assert int(np.argmin(l)) == 2, f"the minimum must be at 1.6e9 (index 2), got index {int(np.argmin(l))}"
