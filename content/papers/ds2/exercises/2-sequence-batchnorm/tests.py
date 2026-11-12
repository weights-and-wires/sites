import numpy as np
from submission import sequence_batchnorm, framewise_batchnorm


def make(seed=0, B=3, T=5, d=4):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=2.0, scale=3.0, size=(B, T, d))


def test_valid_positions_have_zero_mean_and_unit_variance_per_unit():
    a = make()
    lengths = np.array([5, 3, 4])
    out = sequence_batchnorm(a, lengths, np.ones(4), np.zeros(4), eps=1e-12)
    mask = np.arange(5)[None, :] < lengths[:, None]
    valid = out[mask]
    assert np.allclose(valid.mean(axis=0), 0, atol=1e-8), f"expected mean 0 per unit, got {valid.mean(axis=0)}"
    assert np.allclose(valid.var(axis=0), 1, atol=1e-6), f"expected variance 1 per unit, got {valid.var(axis=0)}"


def test_padded_positions_are_zero():
    a = make()
    lengths = np.array([5, 3, 4])
    out = sequence_batchnorm(a, lengths, np.ones(4), np.full(4, 7.0))
    assert np.all(out[1, 3:] == 0), f"expected zeros at padded positions of utterance 1, got {out[1, 3:]}"
    assert np.all(out[2, 4:] == 0), f"expected zeros at padded positions of utterance 2, got {out[2, 4:]}"


def test_padding_values_do_not_change_the_result():
    a = make()
    lengths = np.array([5, 3, 4])
    want = sequence_batchnorm(a, lengths, np.ones(4), np.zeros(4))
    b = a.copy()
    b[1, 3:] = 1e6
    b[2, 4:] = -1e6
    got = sequence_batchnorm(b, lengths, np.ones(4), np.zeros(4))
    assert np.allclose(got, want), "expected padded values to be ignored, but the output changed"


def test_page_example():
    # Utterance 0 has 3 frames and utterance 1 has 2. Valid values: 1, 2, 3, 4, 6.
    a = np.array([[[1.0], [2.0], [3.0]], [[4.0], [6.0], [0.0]]])
    out = sequence_batchnorm(a, np.array([3, 2]), np.ones(1), np.zeros(1), eps=0.0)
    mean, var = 3.2, 2.96
    want = np.array([[[1 - mean], [2 - mean], [3 - mean]], [[4 - mean], [6 - mean], [0.0]]]) / np.sqrt(var)
    assert np.allclose(out, want, atol=1e-6), f"expected {want.ravel()}, got {out.ravel()}"


def test_units_are_normalized_separately():
    a = make()
    a[:, :, 1] *= 100.0
    out = sequence_batchnorm(a, np.array([5, 5, 5]), np.ones(4), np.zeros(4), eps=1e-12)
    assert np.allclose(out.reshape(-1, 4).std(axis=0), 1, atol=1e-6), f"expected unit std in every unit, got {out.reshape(-1, 4).std(axis=0)}"


def test_gamma_and_beta_scale_and_shift_each_unit():
    a = make()
    lengths = np.array([5, 4, 2])
    base = sequence_batchnorm(a, lengths, np.ones(4), np.zeros(4))
    gamma = np.array([1.0, 2.0, 3.0, 4.0])
    beta = np.array([0.5, -1.0, 0.0, 10.0])
    got = sequence_batchnorm(a, lengths, gamma, beta)
    mask = np.arange(5)[None, :] < lengths[:, None]
    want = np.where(mask[:, :, None], base * gamma + beta, 0.0)
    assert np.allclose(got, want), f"expected gamma * normalized + beta at valid positions; max error {np.abs(got - want).max():.3g}"


def test_eps_goes_inside_the_square_root():
    a = np.array([[[0.0], [2.0]]])  # mean 1, variance 1
    out = sequence_batchnorm(a, np.array([2]), np.ones(1), np.zeros(1), eps=3.0)
    want = np.array([[[-1 / 2.0], [1 / 2.0]]])  # sqrt(1 + 3) = 2
    assert np.allclose(out, want), f"expected {want.ravel()}, got {out.ravel()}"


def test_variance_divides_by_the_count():
    a = np.array([[[0.0], [2.0]]])
    out = sequence_batchnorm(a, np.array([2]), np.ones(1), np.zeros(1), eps=0.0)
    assert np.allclose(out, [[[-1.0], [1.0]]]), f"expected [-1, 1] from the biased variance, got {out.ravel()}"


def test_framewise_uses_one_time_step_at_a_time():
    a = np.array([[[1.0], [10.0]], [[3.0], [30.0]]])  # B=2, T=2, d=1
    out = framewise_batchnorm(a, np.ones(1), np.zeros(1), eps=0.0)
    want = np.array([[[-1.0], [-1.0]], [[1.0], [1.0]]])
    assert np.allclose(out, want), f"expected {want.ravel()}, got {out.ravel()}"


def test_framewise_with_one_utterance_returns_beta():
    a = make(B=1)
    out = framewise_batchnorm(a, np.ones(4), np.full(4, 0.25))
    assert np.allclose(out, 0.25), f"expected beta everywhere (variance 0), got min {out.min()} and max {out.max()}"
    seq = sequence_batchnorm(a, np.array([5]), np.ones(4), np.full(4, 0.25))
    assert seq.std() > 0.5, f"expected sequence-wise output to keep its variation for one utterance, got std {seq.std():.3f}"


def test_the_two_methods_differ_on_the_same_input():
    a = make(seed=5, B=4, T=6)
    s = sequence_batchnorm(a, np.array([6, 6, 6, 6]), np.ones(4), np.zeros(4))
    f = framewise_batchnorm(a, np.ones(4), np.zeros(4))
    assert not np.allclose(s, f, atol=1e-3), "expected different outputs: sequence-wise pools all time steps, frame-wise does not"
    assert np.allclose(f.mean(axis=0), 0, atol=1e-6), f"expected every time step to have batch mean 0, got {f.mean(axis=0)[:2]}"
