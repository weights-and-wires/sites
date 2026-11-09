import numpy as np

from submission import rmsnorm, mtp_combine, mtp_loss, mtp_total


def test_rmsnorm():
    x = np.array([[3.0, 4.0], [0.0, 2.0]])
    got = rmsnorm(x, eps=0.0)
    want = x / np.array([[np.sqrt(12.5)], [np.sqrt(2.0)]])
    assert np.allclose(got, want), f"expected {want}, got {got}"
    assert np.allclose(np.sqrt((got**2).mean(axis=1)), 1.0), "each row must have root mean square 1"


def test_combine_shape_and_values():
    h = np.array([[1.0, 1.0, 1.0]])
    e = np.array([[2.0, 0.0, 0.0]])
    M = np.zeros((3, 6))
    M[0, 0] = 1.0  # picks normalized h[0]
    M[1, 3] = 1.0  # picks normalized e[0]
    M[2, 4] = 1.0  # picks normalized e[1]
    got = mtp_combine(h, e, M)
    en = 2.0 / np.sqrt(4.0 / 3.0)
    want = np.array([[1.0, en, 0.0]])
    assert got.shape == (1, 3), f"expected shape (1, 3), got {got.shape}"
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_combine_order_h_first_and_normalizes_each_part_separately():
    h = np.array([[10.0, 0.0]])
    e = np.array([[0.0, 0.1]])
    M = np.eye(2, 4)  # takes the first two entries of the concatenation: the h half
    got = mtp_combine(h, e, M)
    want = rmsnorm(h)
    assert np.allclose(got, want), f"the first half of the concatenation is RMSNorm(h): expected {want}, got {got}"
    M2 = np.eye(4)[2:]  # the e half
    got2 = mtp_combine(h, e, M2)
    want2 = rmsnorm(e)
    assert np.allclose(got2, want2), f"the second half is RMSNorm(emb) with its own scale: expected {want2}, got {got2}"


def test_loss_uniform_and_divisor_is_T():
    T, V, k = 6, 100, 1
    tokens = np.arange(T + 1) % V
    got = mtp_loss(np.zeros((T - k, V)), tokens, k)
    want = (T - k) / T * np.log(V)
    assert abs(got - want) < 1e-9, f"expected {T - k}/{T} * ln {V} = {want:.6f}, got {got:.6f}"


def test_loss_target_alignment():
    T, V, k = 5, 7, 2
    tokens = np.array([0, 1, 2, 3, 4, 5])  # t_1..t_6
    logits = np.full((T - k, V), -30.0)
    for r in range(T - k):
        logits[r, tokens[r + k + 1]] = 30.0  # position i = r+1 predicts t_{i+k+1}
    good = mtp_loss(logits, tokens, k)
    assert good < 1e-6, f"logits that favor t_(i+k+1) must give about 0, got {good:.6f}"
    shifted = np.full((T - k, V), -30.0)
    for r in range(T - k):
        shifted[r, tokens[r + k]] = 30.0  # off by one
    bad = mtp_loss(shifted, tokens, k)
    assert bad > 20, f"favoring t_(i+k) is the wrong target and must be penalized, got {bad:.3f}"


def test_loss_depth_changes_targets():
    tokens = np.array([0, 1, 2, 3, 4, 5, 6])
    T = 6
    V = 7
    logits1 = np.full((T - 1, V), -30.0)
    for r in range(T - 1):
        logits1[r, tokens[r + 2]] = 30.0
    l_right = mtp_loss(logits1, tokens, 1)
    l_wrong = mtp_loss(logits1[:-1], tokens, 2)
    assert l_right < 1e-6, f"depth 1 targets t_(i+2): expected about 0, got {l_right:.6f}"
    assert l_wrong > 10, f"the same logits at depth 2 point at the wrong tokens: expected a large loss, got {l_wrong:.3f}"


def test_loss_known_value():
    logits = np.array([[0.0, 1.0], [2.0, 0.0]])
    tokens = np.array([0, 0, 1, 0])  # T = 3, k = 1: rows predict t_3 = 1, t_4 = 0
    p0 = -np.log(np.exp(1.0) / (1 + np.exp(1.0)))
    p1 = -np.log(np.exp(2.0) / (1 + np.exp(2.0)))
    want = (p0 + p1) / 3
    got = mtp_loss(logits, tokens, 1)
    assert abs(got - want) < 1e-9, f"expected {want:.6f}, got {got:.6f}"


def test_loss_stable_for_large_logits():
    got = mtp_loss(np.array([[1000.0, 0.0]]), np.array([0, 0, 1]), 1)
    assert np.isfinite(got) and abs(got - 1000.0 / 2) < 1e-6, f"expected 500.0 (target logit is 1000 below), got {got}"


def test_total():
    got = mtp_total([2.0, 3.0], 0.3)
    assert abs(got - 0.75) < 1e-12, f"expected 0.3 * mean(2, 3) = 0.75, got {got}"
    got1 = mtp_total([2.0], 0.1)
    assert abs(got1 - 0.2) < 1e-12, f"expected 0.2 for one depth, got {got1}"
