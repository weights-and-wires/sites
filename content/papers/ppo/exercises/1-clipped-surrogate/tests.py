import numpy as np
from submission import clip_grad_logp, clipped_surrogate

# One sample at a time: log(r) with logp_old = 0.
def single(r, a, eps=0.2):
    return clipped_surrogate(np.array([np.log(r)]), np.array([0.0]), np.array([a]), eps)


def test_table_from_the_page():
    cases = [(1.0, 1.1, 1.1), (1.0, 1.5, 1.2), (1.0, 0.5, 0.5), (-1.0, 0.9, -0.9), (-1.0, 0.5, -0.8), (-1.0, 1.5, -1.5)]
    for a, r, want in cases:
        got = single(r, a)
        assert abs(got - want) < 1e-9, f"A={a}, r={r}: expected {want}, got {got}"


def test_batch_mean():
    ratio = np.array([1.3, 0.7, 1.1, 0.5])
    adv = np.array([2.0, 1.0, -1.0, -2.0])
    logp_old = np.log(np.array([0.25, 0.4, 0.5, 0.3]))
    got = clipped_surrogate(logp_old + np.log(ratio), logp_old, adv, 0.2)
    assert abs(got - 0.1) < 1e-9, f"expected 0.1 (the mean, not the sum 0.4), got {got}"


def test_uses_log_probabilities():
    # Ratio is exp(difference), not the difference itself.
    got = clipped_surrogate(np.array([-0.5]), np.array([-1.0]), np.array([1.0]), 10.0)
    want = float(np.exp(0.5))
    assert abs(got - want) < 1e-9, f"expected exp(0.5)={want}, got {got}"


def test_eps_is_used():
    got = single(1.5, 1.0, eps=0.5)
    assert abs(got - 1.5) < 1e-9, f"eps=0.5 keeps r=1.5 unclipped; expected 1.5, got {got}"
    got = single(1.5, 1.0, eps=0.1)
    assert abs(got - 1.1) < 1e-9, f"eps=0.1 clips r=1.5 to 1.1; expected 1.1, got {got}"


def test_gradient_at_ratio_one():
    adv = np.array([1.0, -2.0, 0.5])
    lp = np.log(np.array([0.2, 0.3, 0.5]))
    g = clip_grad_logp(lp, lp, adv, 0.2)
    assert np.allclose(g, adv / 3), f"at r=1 the gradient is A/N = {adv / 3}, got {g}"


def test_gradient_zero_when_clipped():
    adv = np.array([1.0, -1.0])
    logp_old = np.zeros(2)
    # A>0 with r=1.5, A<0 with r=0.5: the clipped term is the min, so no gradient.
    g = clip_grad_logp(np.log(np.array([1.5, 0.5])), logp_old, adv, 0.2)
    assert np.allclose(g, 0.0), f"expected [0, 0], got {g}"


def test_gradient_kept_on_the_pessimistic_side():
    adv = np.array([1.0, -1.0])
    logp_old = np.zeros(2)
    # A>0 with r=0.5 and A<0 with r=1.5: the unclipped term is the min, gradient stays.
    g = clip_grad_logp(np.log(np.array([0.5, 1.5])), logp_old, adv, 0.2)
    assert np.allclose(g, [0.25, -0.75]), f"expected [0.25, -0.75] (r*A/N), got {g}"


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    ratios = np.array([1.35, 0.65, 1.05, 0.9, 1.6, 0.4, 1.12, 0.75])
    adv = rng.normal(size=8)
    logp_old = rng.normal(size=8)
    logp_new = logp_old + np.log(ratios)
    g = clip_grad_logp(logp_new, logp_old, adv, 0.2)
    h = 1e-6
    for i in range(8):
        up, dn = logp_new.copy(), logp_new.copy()
        up[i] += h
        dn[i] -= h
        fd = (clipped_surrogate(up, logp_old, adv, 0.2) - clipped_surrogate(dn, logp_old, adv, 0.2)) / (2 * h)
        assert abs(fd - g[i]) < 1e-6, f"sample {i} (r={ratios[i]}, A={adv[i]:.3f}): finite difference {fd}, analytic {g[i]}"
