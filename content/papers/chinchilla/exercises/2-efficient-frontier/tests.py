import numpy as np
from submission import frontier

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(n, d):
    return E + A / n**ALPHA + B / d**BETA


def test_uses_the_whole_budget():
    for c in [1e19, 1e21, 5.76e23]:
        n, d = frontier(c)
        assert abs(6 * n * d / c - 1) < 1e-9, f"C={c:.2e}: expected 6*N*D = C, got {6 * n * d:.4e}"


def test_known_values():
    n, d = frontier(1e21)
    assert abs(n / 1.824e9 - 1) < 2e-3, f"C=1e21: expected N_opt = 1.824e9, got {n:.4e}"
    assert abs(d / 9.136e10 - 1) < 2e-3, f"C=1e21: expected D_opt = 9.136e10, got {d:.4e}"
    n, d = frontier(5.76e23)
    assert abs(n / 3.219e10 - 1) < 2e-3, f"C=5.76e23: expected N_opt = 3.219e10, got {n:.4e}"


def test_is_a_minimum_of_the_loss():
    for c in [1e19, 1e22]:
        n, d = frontier(c)
        best = loss(n, d)
        for f in [0.7, 0.9, 1.1, 1.4]:
            n2 = n * f
            other = loss(n2, c / (6 * n2))
            assert other > best, f"C={c:.0e}: moving N by x{f} gave {other:.5f}, which is not above the frontier loss {best:.5f}"


def test_exponents():
    n1, d1 = frontier(1e21)
    n2, d2 = frontier(1e22)
    a = np.log10(n2 / n1)
    b = np.log10(d2 / d1)
    assert abs(a - 0.4516) < 2e-3, f"N_opt exponent a: expected 0.4516 (beta/(alpha+beta)), got {a:.4f}"
    assert abs(b - 0.5484) < 2e-3, f"D_opt exponent b: expected 0.5484 (alpha/(alpha+beta)), got {b:.4f}"


def test_vectorised():
    n, d = frontier(np.array([1e19, 1e21]))
    assert np.shape(n) == (2,) and np.shape(d) == (2,), f"expected two values each, got shapes {np.shape(n)} and {np.shape(d)}"
