import numpy as np
from submission import rope, decoupled_scores


def test_rope_hand_values():
    got = rope(np.array([[1.0, 0.0]]), np.array([1]))
    want = np.array([[np.cos(1.0), np.sin(1.0)]])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_rope_pairs_are_adjacent():
    # d = 4: pair 0 is (x0, x1) with theta 1; pair 1 is (x2, x3) with theta 0.01.
    got = rope(np.array([[1.0, 0.0, 1.0, 0.0]]), np.array([2]))
    want = np.array([[np.cos(2.0), np.sin(2.0), np.cos(0.02), np.sin(0.02)]])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_rope_preserves_norm_and_position_zero():
    x = np.random.default_rng(0).normal(size=(5, 8))
    assert np.allclose(rope(x, np.zeros(5)), x), "position 0 must be the identity"
    assert np.allclose(np.linalg.norm(rope(x, np.arange(5)), axis=-1), np.linalg.norm(x, axis=-1)), "RoPE must preserve norms"


def test_hand_example():
    # content score 5, rope part cos(1) between positions 1 and 0, d_h = d_hR = 2.
    qC = np.array([[[1.0, 2.0]], [[1.0, 2.0]]])
    kC = np.array([[[3.0, 1.0]], [[3.0, 1.0]]])
    qR = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
    kR = np.array([[1.0, 0.0], [1.0, 0.0]])
    s = decoupled_scores(qC, kC, qR, kR, 2, 2)
    want = (5.0 + np.cos(1.0)) / 2.0
    assert s.shape == (1, 2, 2), f"expected shape (1, 2, 2), got {s.shape}"
    assert abs(s[0, 1, 0] - want) < 1e-9, f"score [0, 1, 0] expected {want:.4f}, got {s[0, 1, 0]:.4f}"
    want00 = (5.0 + 1.0) / 2.0
    assert abs(s[0, 0, 0] - want00) < 1e-9, f"score [0, 0, 0] expected {want00:.4f}, got {s[0, 0, 0]:.4f}"


def rand(T=5, n_h=3, d_h=4, d_hR=6, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(T, n_h, d_h)), r.normal(size=(T, n_h, d_h)), r.normal(size=(T, n_h, d_hR)), r.normal(size=(T, d_hR))


def test_content_part_has_no_position():
    qC, kC, qR, kR = rand()
    s = decoupled_scores(qC, kC, np.zeros_like(qR), np.zeros_like(kR), 4, 6)
    want = np.einsum("tid,jid->itj", qC, kC) / np.sqrt(10)
    assert np.allclose(s, want), f"with zero RoPE parts expected the plain content scores {want[0, 0]}, got {s[0, 0]}"


def test_rope_part_is_relative():
    # Same vectors at every position: the RoPE part then depends only on t - j.
    T, n_h, d_hR = 6, 2, 4
    r = np.random.default_rng(1)
    q1, k1 = r.normal(size=(n_h, d_hR)), r.normal(size=(d_hR,))
    qR = np.tile(q1, (T, 1, 1))
    kR = np.tile(k1, (T, 1))
    z = np.zeros((T, n_h, 1))
    s = decoupled_scores(z, z, qR, kR, 1, d_hR)
    for i in range(n_h):
        for off in range(-2, 3):
            vals = [s[i, t, t - off] for t in range(T) if 0 <= t - off < T]
            assert np.allclose(vals, vals[0]), f"head {i}, offset {off}: scores should be equal, got {vals}"


def test_key_rope_is_shared_across_heads():
    qC, kC, qR, kR = rand(seed=2)
    z = np.zeros_like(qC)
    qR2 = np.tile(qR[:, :1], (1, qR.shape[1], 1))
    s = decoupled_scores(z, z, qR2, kR, 4, 6)
    assert np.allclose(s[0], s[1]) and np.allclose(s[1], s[2]), "equal queries with a shared key must give equal scores in every head"


def test_scale_uses_both_widths():
    qC, kC, qR, kR = rand(seed=3)
    s = decoupled_scores(qC, kC, qR, kR, 4, 6)
    raw = decoupled_scores(qC, kC, qR, kR, 1, 0)  # divides by sqrt(1)
    assert np.allclose(s * np.sqrt(10), raw), f"expected a divisor of sqrt(4 + 6), got ratio {raw[0, 0, 0] / s[0, 0, 0]:.4f}"
