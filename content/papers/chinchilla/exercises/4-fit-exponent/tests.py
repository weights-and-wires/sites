import numpy as np
from submission import fit_exponent, grow

N = np.array([400e6, 1e9, 10e9, 67e9, 175e9, 280e9, 520e9, 1e12, 10e12])
C = np.array([1.92e19, 1.21e20, 1.23e22, 5.76e23, 3.85e24, 9.90e24, 3.43e25, 1.27e26, 1.30e28])
D = np.array([8.0e9, 20.2e9, 205.1e9, 1.5e12, 3.7e12, 5.9e12, 11.0e12, 21.2e12, 216.2e12])


def test_recovers_a_known_exponent():
    c = np.logspace(18, 24, 7)
    p = fit_exponent(c, 3.0 * c**0.37)
    assert abs(p - 0.37) < 1e-9, f"expected 0.37, got {p}"
    q = fit_exponent(c, 5e4 * c**-0.05)
    assert abs(q + 0.05) < 1e-9, f"a decreasing law has a negative exponent: expected -0.05, got {q}"


def test_table_3_model_size_exponent():
    a = fit_exponent(C, N)
    assert abs(a - 0.4981) < 2e-3, f"Table 3, N_opt vs C: expected 0.498, got {a:.4f}"


def test_table_3_token_exponent():
    b = fit_exponent(C, D)
    assert abs(b - 0.5019) < 2e-3, f"Table 3, D_opt vs C: expected 0.502, got {b:.4f}"


def test_grow_conserves_the_compute_ratio():
    n1, d1 = grow(1e9, 20e9, 100.0, 0.5)
    assert abs(6 * n1 * d1 / (6 * 1e9 * 20e9) - 100.0) < 1e-9, f"expected the compute to grow x100, got x{n1 * d1 / (1e9 * 20e9):.4f}"
    assert abs(n1 / 1e10 - 1) < 1e-9, f"a=0.5 and k=100: expected N1 = 1e10, got {n1:.4e}"
    assert abs(d1 / 2e11 - 1) < 1e-9, f"a=0.5 and k=100: expected D1 = 2e11, got {d1:.4e}"


def test_kaplan_versus_hoffmann_split():
    k = 1000.0
    nk, dk = grow(1e9, 20e9, k, 0.73)
    nh, dh = grow(1e9, 20e9, k, 0.50)
    assert abs(dk / 20e9 - 1000 ** 0.27) < 1e-6 * 1000 ** 0.27, f"Kaplan: expected D1/D0 = {1000 ** 0.27:.3f}, got {dk / 20e9:.3f}"
    assert abs(dh / 20e9 - 1000 ** 0.5) < 1e-6 * 1000 ** 0.5, f"Hoffmann: expected D1/D0 = {1000 ** 0.5:.3f}, got {dh / 20e9:.3f}"
    assert nk > nh and dk < dh, f"Kaplan grows the model faster and the data slower; got N {nk:.3e} vs {nh:.3e}, D {dk:.3e} vs {dh:.3e}"
