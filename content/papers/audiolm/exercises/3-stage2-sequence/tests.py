import numpy as np
from submission import build_stage2

K, N, Q, QP = 8, 16, 3, 2
Z = np.array([3, 3, 5])
Y = np.array([[1, 2, 9], [4, 5, 9], [6, 7, 9], [8, 9, 9], [10, 11, 9], [12, 13, 9]])
WANT = [3, 3, 5, 9, 26, 12, 29, 14, 31, 16, 33, 18, 35, 20, 37]


def test_page_example():
    seq, _ = build_stage2(Z, Y, QP, K, N)
    assert np.asarray(seq).tolist() == WANT, f"expected {WANT}, got {np.asarray(seq).tolist()}"


def test_length_uses_only_coarse_layers():
    seq, mask = build_stage2(Z, Y, QP, K, N)
    assert len(seq) == 3 + 6 * 2, f"expected length T_S + T_A * Q' = 15, got {len(seq)}"
    assert len(mask) == len(seq), f"mask length must match: expected {len(seq)}, got {len(mask)}"


def test_loss_mask():
    _, mask = build_stage2(Z, Y, QP, K, N)
    mask = np.asarray(mask)
    assert mask.dtype == bool, f"expected a boolean mask, got {mask.dtype}"
    want = [0, 0, 0] + [1] * 12
    assert mask.astype(int).tolist() == want, f"expected {want}, got {mask.astype(int).tolist()}"


def test_row_major_interleaving():
    seq, _ = build_stage2(Z, Y, QP, K, N)
    acoustic = np.asarray(seq)[3:]
    steps = acoustic.reshape(-1, QP)
    assert (steps[:, 0] - K).max() < N and (steps[:, 1] - K).min() >= N, "layer 0 ids must be in [K, K+N) and layer 1 ids in [K+N, K+2N) at every step"
    assert (steps[:, 0] - K).tolist() == Y[:, 0].tolist(), f"layer 0 should follow step order, expected {Y[:, 0].tolist()}, got {(steps[:, 0] - K).tolist()}"


def test_vocabularies_do_not_collide():
    rng = np.random.default_rng(0)
    z = rng.integers(0, 1024, size=20)
    y = rng.integers(0, 1024, size=(40, 12))
    seq, mask = build_stage2(z, y, 4, 1024, 1024)
    seq, mask = np.asarray(seq), np.asarray(mask)
    assert seq[:20].max() < 1024, f"semantic ids must stay below K, got max {seq[:20].max()}"
    assert seq[20:].min() >= 1024 and seq.max() < 1024 + 4 * 1024, f"acoustic ids must lie in [1024, 5120), got [{seq[20:].min()}, {seq.max()}]"
    assert len(seq) == 20 + 40 * 4, f"expected {20 + 160}, got {len(seq)}"


def test_other_qprime():
    seq, mask = build_stage2(Z, Y, 3, K, N)
    assert len(seq) == 3 + 6 * 3, f"Q'=3 should give length 21, got {len(seq)}"
    assert np.asarray(mask).sum() == 18, f"expected 18 predicted tokens, got {np.asarray(mask).sum()}"
