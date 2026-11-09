import numpy as np
from submission import loss_nd, loss_n_only, overfit_penalty, min_tokens


def test_known_values():
    got = [float(loss_nd(1e9, d)) for d in (1e8, 1e9, 1e10)]
    want = [3.4846, 2.7943, 2.4197]
    assert np.allclose(got, want, atol=1e-3), f"expected {want}, got {got}"


def test_infinite_data_limit_is_loss_n():
    got, want = float(loss_nd(1e9, 1e40)), float(loss_n_only(1e9))
    assert abs(got - want) < 1e-6, f"as D grows, L(N, D) must approach L(N) = {want}, got {got}"
    assert abs(want - 2.3188) < 1e-3, f"expected L(N) = 2.3188 at N = 1e9, got {want}"


def test_infinite_model_limit_is_loss_d():
    got = float(loss_nd(1e60, 1e10))
    want = (1.8e13 / 1e10) ** 0.103
    assert abs(got - want) < 1e-6, f"as N grows, L(N, D) must approach L(D) = {want}, got {got}"


def test_more_data_or_parameters_never_hurt():
    assert loss_nd(1e9, 1e10) < loss_nd(1e9, 1e9), "more data at fixed N must lower the loss"
    assert loss_nd(1e9, 1e10) < loss_nd(1e8, 1e10), "more parameters at fixed D must lower the loss"


def test_works_on_arrays():
    n = np.array([1e8, 1e9])
    got = loss_nd(n, 1e10)
    assert np.shape(got) == (2,), f"expected shape (2,), got {np.shape(got)}"
    assert np.allclose(got, [loss_nd(1e8, 1e10), loss_nd(1e9, 1e10)]), f"array result must match scalar calls, got {got}"


def test_overfit_penalty_values():
    got = [float(overfit_penalty(1e9, d)) for d in (1e8, 1e10, 2.2e10)]
    want = [0.5028, 0.0435, 0.0218]
    assert np.allclose(got, want, atol=2e-3), f"expected {want}, got {got}"


def test_overfit_penalty_depends_on_one_ratio():
    # Equation 4.3: deltaL depends on N and D only through N**(alpha_N/alpha_D) / D.
    r = 0.076 / 0.103
    a = float(overfit_penalty(1e8, 1e9))
    k = 10 ** (1 / r)  # multiplying N by k multiplies N**r by 10
    b = float(overfit_penalty(1e8 * k, 1e10))
    assert abs(a - b) < 1e-9, f"scaling D by 10 and N by 10**(1/r) must leave delta L unchanged: {a} vs {b}"


def test_min_tokens():
    got = [float(min_tokens(n)) for n in (1e8, 1e9)]
    want = [4.159e9, 2.285e10]
    assert np.allclose(got, want, rtol=1e-3), f"expected {want}, got {got}"
    assert abs(float(min_tokens(8e9) / min_tokens(1e9)) - 8**0.74) < 1e-9, "8x the parameters needs 8**0.74 (about 4.66x) the tokens"
