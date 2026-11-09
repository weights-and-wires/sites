import numpy as np
from submission import dqn_loss, td_targets

gamma = 0.9
q = np.array([[0.5, 0.2], [0.5, 0.9], [0.1, 0.3]])
a = np.array([1, 0, 0])
r = np.array([0.0, 1.0, -1.0])
q_next = np.array([[0.4, 0.8], [7.0, 7.0], [0.2, 0.1]])
done = np.array([0.0, 1.0, 0.0])


def test_targets_on_page_example():
    y = np.asarray(td_targets(r, q_next, done, gamma))
    want = np.array([0.72, 1.0, -0.82])
    assert y.shape == (3,), f"expected shape (3,), got {y.shape}"
    assert np.allclose(y, want), f"expected {want}, got {y}"


def test_terminal_ignores_next_q():
    big = q_next.copy()
    big[1] = [1e6, 1e6]
    y = np.asarray(td_targets(r, big, done, gamma))
    assert np.isclose(y[1], 1.0), f"a terminal row must give y = r = 1.0, got {y[1]}"


def test_max_is_over_actions():
    qn = np.array([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]])
    y = np.asarray(td_targets(np.zeros(2), qn, np.zeros(2), 1.0))
    assert np.allclose(y, [5.0, 4.0]), f"expected the per-row maximum [5.0, 4.0], got {y}"


def test_negative_next_values():
    qn = np.array([[-3.0, -1.0]])
    y = np.asarray(td_targets(np.array([0.0]), qn, np.array([0.0]), 0.5))
    assert np.allclose(y, [-0.5]), f"expected 0.5 * max(-3, -1) = -0.5, got {y}"


def test_gamma_scales_only_the_bootstrap():
    y = np.asarray(td_targets(np.array([1.0]), np.array([[2.0, 2.0]]), np.array([0.0]), 0.5))
    assert np.allclose(y, [2.0]), f"expected 1 + 0.5 * 2 = 2.0, got {y}"


def test_loss_on_page_example():
    y = td_targets(r, q_next, done, gamma)
    got = dqn_loss(q, a, y)
    assert np.isclose(got, 0.455533, atol=1e-4), f"expected 0.4555, got {got}"


def test_loss_uses_the_taken_action():
    qq = np.array([[10.0, 0.0], [0.0, 10.0]])
    aa = np.array([1, 0])
    got = dqn_loss(qq, aa, np.array([0.0, 0.0]))
    assert np.isclose(got, 0.0), f"Q(phi, a) are both 0 here, so the loss is 0; got {got}"
    got = dqn_loss(qq, np.array([0, 1]), np.array([0.0, 0.0]))
    assert np.isclose(got, 100.0), f"Q(phi, a) are both 10 here, so the loss is 100; got {got}"


def test_loss_is_a_mean_of_squares():
    qq = np.array([[1.0, 0.0], [0.0, 0.0]])
    got = dqn_loss(qq, np.array([0, 0]), np.array([-1.0, 1.0]))
    assert np.isclose(got, 2.5), f"errors 2 and 1 give (4 + 1) / 2 = 2.5, got {got}"
