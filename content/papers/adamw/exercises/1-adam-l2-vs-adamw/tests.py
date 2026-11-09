import numpy as np
from submission import adam_l2_step, adamw_step

Z = np.zeros(1)


def test_lambda_zero_gives_plain_adam_for_both():
    theta, g = np.array([0.7, -1.2]), np.array([0.3, 2.0])
    m, v = np.array([0.1, -0.2]), np.array([0.05, 0.4])
    ref = theta - 1e-3 * (0.9 * m + 0.1 * g) / (1 - 0.9**7) / (np.sqrt((0.999 * v + 0.001 * g * g) / (1 - 0.999**7)) + 1e-8)
    for fn in (adam_l2_step, adamw_step):
        out, _, _ = fn(theta, g, m, v, 7, 0.0)
        assert np.allclose(out, ref, atol=1e-12), f"{fn.__name__}: expected plain Adam {ref}, got {out}"


def test_adamw_with_zero_gradient_shrinks_by_one_minus_lambda():
    theta = np.array([2.0, -4.0])
    out, m, v = adamw_step(theta, np.zeros(2), np.zeros(2), np.zeros(2), 1, 0.1)
    want = 0.9 * theta
    assert np.allclose(out, want, atol=1e-12), f"expected (1 - lam) * theta = {want}, got {out}"
    assert np.allclose(m, 0) and np.allclose(v, 0), f"the decay must not enter the moments, got m={m}, v={v}"


def test_adamw_decay_is_not_scaled_by_alpha():
    out_a, _, _ = adamw_step(np.array([1.0]), Z, Z, Z, 1, 0.05, alpha=1e-3)
    out_b, _, _ = adamw_step(np.array([1.0]), Z, Z, Z, 1, 0.05, alpha=1e-1)
    assert np.allclose(out_a, 0.95) and np.allclose(out_b, 0.95), f"expected 0.95 for both alphas, got {out_a} and {out_b}"


def test_adamw_moments_see_only_the_loss_gradient():
    g = np.array([0.5])
    _, m, v = adamw_step(np.array([10.0]), g, Z, Z, 1, 0.3)
    assert np.allclose(m, 0.05) and np.allclose(v, 0.00025), f"expected m=0.05, v=0.00025, got m={m}, v={v}"


def test_adam_l2_puts_the_decay_inside_the_moments():
    theta, g = np.array([2.0]), np.array([0.5])
    _, m, v = adam_l2_step(theta, g, Z, Z, 1, 0.1)
    gl = 0.5 + 0.1 * 2.0
    assert np.allclose(m, 0.1 * gl) and np.allclose(v, 0.001 * gl * gl), f"expected m={0.1 * gl}, v={0.001 * gl**2}, got m={m}, v={v}"


def test_adam_l2_zero_gradient_step_ignores_lambda():
    # Adam normalizes lam * theta by its own size, so the step is about alpha for any lam.
    steps = []
    for lam in (0.1, 0.01, 0.001):
        out, _, _ = adam_l2_step(np.array([1.0]), Z, Z, Z, 1, lam)
        steps.append(float(1.0 - out[0]))
    assert np.allclose(steps, 1e-3, atol=1e-6), f"expected a step of about alpha = 0.001 for every lam, got {steps}"


def test_l2_decay_shrinks_with_gradient_scale_but_adamw_does_not():
    # theta = 1, same lam, loss gradients of size 1 and 100 after a long history (m = g, v = g^2).
    got = {}
    for s in (1.0, 100.0):
        g = np.array([s])
        m, v = g.copy(), g * g
        t = 5000
        base, _, _ = adam_l2_step(np.array([1.0]), g, m, v, t, 0.0)
        with_l2, _, _ = adam_l2_step(np.array([1.0]), g, m, v, t, 0.1)
        w0, _, _ = adamw_step(np.array([1.0]), g, m, v, t, 0.1)
        got[s] = (float(base[0] - with_l2[0]), float(base[0] - w0[0]))
    l2_small, l2_big = got[1.0][0], got[100.0][0]
    assert l2_small > 20 * l2_big, f"expected the L2 effect to drop about 100x when the gradient is 100x larger, got {l2_small:.3e} and {l2_big:.3e}"
    assert np.isclose(got[1.0][1], got[100.0][1], rtol=1e-3), f"expected the same AdamW effect, got {got[1.0][1]:.3e} and {got[100.0][1]:.3e}"
    assert np.isclose(got[1.0][1], 0.1, rtol=1e-6), f"expected the AdamW decay term 0.1 * theta = 0.1, got {got[1.0][1]:.3e}"


def test_inputs_are_not_modified():
    for fn in (adam_l2_step, adamw_step):
        theta, g, m, v = np.ones(3), np.array([1.0, 2.0, 3.0]), np.full(3, 0.5), np.full(3, 0.25)
        copies = [a.copy() for a in (theta, g, m, v)]
        fn(theta, g, m, v, 5, 0.1)
        for name, before, after in zip("theta grad m v".split(), copies, (theta, g, m, v)):
            assert np.array_equal(before, after), f"{fn.__name__} modified {name}: before {before}, after {after}"
