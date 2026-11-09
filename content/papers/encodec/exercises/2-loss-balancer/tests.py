import numpy as np
from submission import balance


def test_first_call_gives_each_loss_its_fraction():
    g1 = np.array([3.0, 4.0])
    g2 = np.array([0.0, 0.02])
    total, ema = balance([g1, g2], [1.0, 3.0], None, R=1.0)
    want = 0.25 * g1 / 5.0 + 0.75 * g2 / 0.02
    assert np.allclose(total, want), f"expected {want}, got {total}"
    assert np.allclose(ema, [5.0, 0.02]), f"the first call stores the raw norms [5, 0.02]; got {ema}"


def test_rescaling_a_loss_changes_nothing():
    g1 = np.array([1.0, -2.0, 0.5])
    g2 = np.array([0.3, 0.1, -0.2])
    a, _ = balance([g1, g2], [1.0, 1.0], None)
    b, _ = balance([1000 * g1, g2], [1.0, 1.0], None)
    assert np.allclose(a, b), f"multiplying one loss by 1000 must not change the balanced gradient; got {a} and {b}"


def test_weights_are_normalized_and_R_scales():
    g1 = np.array([1.0, 0.0])
    g2 = np.array([0.0, 1.0])
    total, _ = balance([g1, g2], [0.1, 0.3], None, R=2.0)
    want = np.array([2.0 * 0.25, 2.0 * 0.75])
    assert np.allclose(total, want), f"fractions are 0.25 and 0.75, R = 2; expected {want}, got {total}"
    same, _ = balance([g1, g2], [1.0, 3.0], None, R=2.0)
    assert np.allclose(total, same), f"only the ratio of the weights matters; got {total} and {same}"


def test_second_call_uses_the_running_average():
    g = np.array([3.0, 4.0])
    _, ema = balance([g], [1.0], None)
    total, ema2 = balance([10 * g], [1.0], ema, beta=0.9)
    want_ema = 0.9 * 5.0 + 0.1 * 50.0
    assert np.allclose(ema2, [want_ema]), f"expected EMA {want_ema}, got {ema2}"
    want = 10 * g / want_ema
    assert np.allclose(total, want), f"the gradient is divided by the updated EMA {want_ema}, not by its own norm; expected {want}, got {total}"
    assert np.linalg.norm(total) > 1.0, f"a sudden jump in scale must show up as a norm above R = 1, not be cancelled; got norm {np.linalg.norm(total)}"


def test_works_on_matrices():
    rng = np.random.default_rng(0)
    gs = [rng.normal(size=(4, 5)) for _ in range(3)]
    total, _ = balance(gs, [1.0, 1.0, 2.0], None)
    want = sum(f * g / np.linalg.norm(g) for f, g in zip([0.25, 0.25, 0.5], gs))
    assert total.shape == (4, 5), f"expected shape (4, 5), got {total.shape}"
    assert np.allclose(total, want), "the norm is the Euclidean norm of the whole array (Frobenius for a matrix)"
