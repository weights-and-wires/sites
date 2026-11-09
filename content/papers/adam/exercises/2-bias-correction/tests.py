import numpy as np
from submission import effective_steps


def test_constant_gradient_gives_alpha_every_step():
    got = effective_steps([1.0] * 10, alpha=0.001)
    assert np.allclose(got, 0.001, atol=1e-9), f"expected 0.001 at every step, got {got}"


def test_uncorrected_first_steps_are_too_large():
    got = effective_steps([1.0] * 10, alpha=0.001, bias_correction=False) / 0.001
    assert abs(got[0] - 3.1623) < 1e-3, f"expected step 1 / alpha = 3.1623, got {got[0]:.4f}"
    assert abs(got[9] - 6.5279) < 1e-3, f"expected step 10 / alpha = 6.5279, got {got[9]:.4f}"


def test_sign_and_scale_of_the_first_step():
    got = effective_steps([-5.0], alpha=0.01)
    assert abs(got[0] + 0.01) < 1e-8, f"expected -0.01 (alpha times sign of the gradient), got {got[0]}"


def test_page_trace():
    got = effective_steps([2.0, -1.0, 1.0], alpha=1e-3) / 1e-3
    want = [1.0, 0.2663, 0.4489]
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {np.round(got, 4)}"


def test_each_moment_uses_its_own_beta():
    got = effective_steps([1.0, 1.0], alpha=1.0, beta1=0.5, beta2=0.9, bias_correction=True)
    assert np.allclose(got, 1.0, atol=1e-7), f"expected [1, 1] for a constant gradient, got {got}"
    raw = effective_steps([1.0], alpha=1.0, beta1=0.5, beta2=0.9, bias_correction=False)
    want = 0.5 / np.sqrt(0.1)
    assert abs(raw[0] - want) < 1e-6, f"expected {want:.4f}, got {raw[0]:.4f}"


def test_sparse_gradient_approaches_the_bound():
    got = effective_steps([0.0] * 5000 + [1.0], alpha=1.0)
    bound = 0.1 / np.sqrt(0.001)
    assert abs(got[-1] - 3.1516) < 1e-3, f"expected 3.1516, got {got[-1]:.4f}"
    assert got[-1] <= bound, f"expected at most the Section 2.1 bound {bound:.4f}, got {got[-1]:.4f}"
