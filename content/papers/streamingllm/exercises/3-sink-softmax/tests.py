import numpy as np

from submission import attend, sink_mass, softmax, softmax_one


def toy():
    x = np.array([6.0, 0.2, -0.1, 0.3, 0.0, -0.2, 0.1, 0.2])
    rng = np.random.default_rng(0)
    V = rng.normal(size=(8, 4))
    V[0] = 0.01 * V[0]
    return x, V


def test_softmax_values():
    got = softmax(np.array([0.0, 0.0, 0.0, 0.0]))
    assert np.allclose(got, 0.25), f"expected 0.25 each, got {got}"
    got = softmax(np.array([1000.0, 1000.0]))
    assert np.allclose(got, [0.5, 0.5]), f"large logits must not overflow: expected [0.5, 0.5], got {got}"


def test_softmax_one_values():
    got = softmax_one(np.array([0.0]))
    assert np.allclose(got, [0.5]), f"exp(0) / (1 + exp(0)) = 0.5, got {got}"
    x = np.array([0.2, -0.1, 0.3])
    want = np.exp(x) / (1 + np.exp(x).sum())
    assert np.allclose(softmax_one(x), want), f"expected {want}, got {softmax_one(x)}"


def test_softmax_one_can_sum_below_one():
    s = softmax_one(np.full(5, -50.0)).sum()
    assert s < 1e-6, f"when every logit is very negative the weights should vanish, sum = {s}"
    s = softmax_one(np.array([1000.0, 999.0])).sum()
    assert 0.999 < s <= 1.0, f"large logits approach softmax: expected about 1, got {s}"


def test_softmax_one_stable():
    got = softmax_one(np.array([-1000.0, -1000.0]))
    assert np.all(np.isfinite(got)) and got.sum() < 1e-6, f"expected finite values near 0, got {got}"
    got = softmax_one(np.array([1000.0, 0.0]))
    assert np.all(np.isfinite(got)), f"expected finite values, got {got}"


def test_sink_mass_toy():
    x, _ = toy()
    got = sink_mass(x, 1)
    assert abs(got - 0.9815) < 1e-3, f"expected 0.9815, got {got:.4f}"
    assert sink_mass(x, 0) == 0.0, f"no sinks, no mass, got {sink_mass(x, 0)}"
    assert abs(sink_mass(x, 8) - 1.0) < 1e-9, f"all tokens carry all the mass, got {sink_mass(x, 8)}"


def test_attend_full_matches_softmax():
    x, V = toy()
    want = softmax(x) @ V
    got = attend(x, V)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_evicting_the_sink_renormalizes():
    x, V = toy()
    want = softmax(x[1:]) @ V[1:]
    got = attend(x, V, evict=1)
    assert np.allclose(got, want), f"the denominator must lose the sink's logit: expected {want}, got {got}"
    assert not np.allclose(got, softmax(x)[1:] @ V[1:]), "dropping the row but keeping the old weights is wrong"


def test_eviction_blows_up_the_output():
    x, V = toy()
    a = np.linalg.norm(attend(x, V))
    b = np.linalg.norm(attend(x, V, evict=1))
    assert a < 0.05, f"with the sink, the output norm should be small: expected < 0.05, got {a:.4f}"
    assert b > 0.5, f"without the sink, the output norm should be large: expected > 0.5, got {b:.4f}"
