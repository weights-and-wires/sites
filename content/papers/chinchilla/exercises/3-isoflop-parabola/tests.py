import numpy as np
from submission import isoflop_minimum

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28
SIZES = np.array([0.4e9, 0.8e9, 1.6e9, 3.2e9, 6.4e9, 12.8e9])


def isoflop(c, sizes):
    return E + A / sizes**ALPHA + B / (c / (6 * sizes)) ** BETA


def test_exact_parabola_in_log_n():
    x0, curv, floor = np.log(2e9), 0.05, 2.3
    losses = curv * (np.log(SIZES) - x0) ** 2 + floor
    n, l = isoflop_minimum(SIZES, losses)
    assert abs(n / 2e9 - 1) < 1e-6, f"expected the vertex at N = 2e9, got {n:.6e}"
    assert abs(l - floor) < 1e-9, f"expected the minimum loss 2.3, got {l}"


def test_parabola_in_n_is_not_the_same():
    # A parabola exact in N (not log N) must not give the log-space vertex.
    losses = 1e-18 * (SIZES - 3e9) ** 2 + 2.3
    n, _ = isoflop_minimum(SIZES, losses)
    assert abs(n / 3e9 - 1) > 0.02, f"the fit must be in ln N; a parabola in N should not give exactly 3e9, got {n:.4e}"


def test_fitted_paper_style_slice():
    n, l = isoflop_minimum(SIZES, isoflop(1e21, SIZES))
    assert abs(n / 1.862e9 - 1) < 2e-3, f"C=1e21: expected N = 1.862e9, got {n:.4e}"
    assert abs(l - 2.3288) < 5e-4, f"C=1e21: expected minimum loss 2.3288, got {l:.4f}"


def test_other_budget_and_scaling():
    sizes = SIZES * 100
    n, l = isoflop_minimum(sizes, isoflop(1e25, sizes))
    assert 1e10 < n < 2e11, f"C=1e25 with sizes 4e10..1.3e12: the vertex should be inside the sampled range, got {n:.4e}"
    n2, _ = isoflop_minimum(SIZES, isoflop(1e21, SIZES))
    n3, _ = isoflop_minimum(SIZES, isoflop(1e22, SIZES))
    assert n3 > n2, f"more compute must move the vertex to a larger model: {n2:.3e} then {n3:.3e}"


def test_returns_plain_numbers():
    n, l = isoflop_minimum(SIZES, isoflop(1e21, SIZES))
    assert np.ndim(n) == 0 and np.ndim(l) == 0, f"expected two scalars, got shapes {np.shape(n)} and {np.shape(l)}"
