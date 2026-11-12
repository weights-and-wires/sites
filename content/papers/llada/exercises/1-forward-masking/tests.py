import numpy as np
from submission import forward_mask

MASK = -1


def test_t_zero_keeps_everything():
    x0 = np.arange(40).reshape(4, 10)
    x_t, masked = forward_mask(x0, 0.0, np.random.default_rng(0), MASK)
    assert np.array_equal(x_t, x0), f"at t = 0 nothing is masked; expected {x0.tolist()}, got {x_t.tolist()}"
    assert not masked.any(), f"at t = 0 the mask must be all False, got {int(masked.sum())} True"


def test_t_one_masks_everything():
    x0 = np.arange(40).reshape(4, 10)
    x_t, masked = forward_mask(x0, 1.0, np.random.default_rng(0), MASK)
    assert (x_t == MASK).all(), f"at t = 1 every token is M; got {x_t.tolist()}"
    assert masked.all(), "at t = 1 the mask must be all True"


def test_masked_positions_hold_mask_id_and_others_are_unchanged():
    x0 = np.arange(1, 201).reshape(10, 20)
    x_t, masked = forward_mask(x0, 0.5, np.random.default_rng(1), MASK)
    assert (x_t[masked] == MASK).all(), "masked positions must hold mask_id"
    assert np.array_equal(x_t[~masked], x0[~masked]), "unmasked positions must keep their original token"


def test_mask_rate_is_t_and_not_one_minus_t():
    x0 = np.zeros((2000, 50), dtype=int)
    for t in (0.2, 0.5, 0.8):
        _, masked = forward_mask(x0, t, np.random.default_rng(2), MASK)
        got = masked.mean()
        assert abs(got - t) < 0.01, f"t={t}: expected a mask rate near {t}, got {got:.4f}"


def test_rows_are_masked_independently():
    x0 = np.zeros((200, 30), dtype=int)
    _, masked = forward_mask(x0, 0.5, np.random.default_rng(3), MASK)
    distinct = len({tuple(r) for r in masked.astype(int)})
    assert distinct > 150, f"each row needs its own mask; expected over 150 distinct rows, got {distinct}"
    col_rates = masked.mean(axis=0)
    assert col_rates.min() > 0.3 and col_rates.max() < 0.7, "every column must be masked about half the time"


def test_does_not_modify_the_input():
    x0 = np.arange(30).reshape(3, 10)
    keep = x0.copy()
    forward_mask(x0, 0.7, np.random.default_rng(4), MASK)
    assert np.array_equal(x0, keep), f"x0 must not change; expected {keep.tolist()}, got {x0.tolist()}"


def test_same_seed_gives_same_mask():
    x0 = np.arange(30).reshape(3, 10)
    a, _ = forward_mask(x0, 0.5, np.random.default_rng(5), MASK)
    b, _ = forward_mask(x0, 0.5, np.random.default_rng(5), MASK)
    assert np.array_equal(a, b), "the same seed must give the same masked sequence"
