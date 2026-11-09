import numpy as np
from submission import expert_capacity, keep_mask, switch_layer

IDX = np.array([0, 0, 0, 0, 1, 2])  # the routing from the page


def test_capacity_values():
    assert expert_capacity(6, 3, 1.0) == 2, f"expected 2, got {expert_capacity(6, 3, 1.0)}"
    assert expert_capacity(6, 3, 1.5) == 3, f"expected 3, got {expert_capacity(6, 3, 1.5)}"
    assert expert_capacity(1024, 8, 1.25) == 160, f"expected 160, got {expert_capacity(1024, 8, 1.25)}"


def test_capacity_rounds_up_and_is_int():
    c = expert_capacity(10, 4, 1.0)  # 2.5 rounds up to 3
    assert c == 3, f"10 tokens over 4 experts at factor 1.0: expected 3, got {c}"
    assert isinstance(c, (int, np.integer)), f"capacity must be an integer, got {type(c)}"


def test_keep_mask_on_the_page_example():
    got = keep_mask(IDX, 3, 2)
    assert got.dtype == bool, f"expected a boolean array, got {got.dtype}"
    assert got.tolist() == [True, True, False, False, True, True], f"capacity 2: expected tokens 2 and 3 dropped, got {got.tolist()}"
    got = keep_mask(IDX, 3, 3)
    assert got.tolist() == [True, True, True, False, True, True], f"capacity 3: expected only token 3 dropped, got {got.tolist()}"


def test_earlier_tokens_win_the_slot():
    got = keep_mask(np.array([1, 1, 1]), 2, 1)
    assert got.tolist() == [True, False, False], f"the first token must keep the only slot, got {got.tolist()}"


def test_capacity_is_per_expert():
    # Tokens alternate between experts, so no expert ever exceeds 2 tokens.
    got = keep_mask(np.array([0, 1, 0, 1, 0, 1]), 2, 3)
    assert got.all(), f"3 tokens per expert fit in capacity 3, got {got.tolist()}"
    got = keep_mask(np.array([0, 1, 0, 1, 0, 1]), 2, 2)
    assert got.tolist() == [True, True, True, True, False, False], f"expected the 3rd token of each expert dropped, got {got.tolist()}"


def test_large_capacity_keeps_everything():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 4, size=50)
    assert keep_mask(idx, 4, 50).all(), "capacity equal to the batch size must keep every token"


def test_switch_layer_kept_and_dropped():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 3))
    W = rng.normal(size=(3, 3, 3))
    gate = np.array([0.8, 0.6, 0.5, 0.7, 0.8, 0.8])
    keep = np.array([True, True, False, False, True, True])
    y = switch_layer(x, IDX, gate, keep, W)
    for t in (2, 3):
        assert np.allclose(y[t], x[t]), f"dropped token {t} must equal its input {x[t]}, got {y[t]}"
    for t in (0, 1, 4, 5):
        want = x[t] + gate[t] * (x[t] @ W[IDX[t]])
        assert np.allclose(y[t], want), f"kept token {t}: expected {want}, got {y[t]}"


def test_dropped_token_is_not_zeroed():
    x = np.ones((2, 2))
    W = np.zeros((1, 2, 2))
    y = switch_layer(x, np.array([0, 0]), np.array([1.0, 1.0]), np.array([True, False]), W)
    assert np.allclose(y, x), f"a zero expert leaves x unchanged, and a dropped token passes through; got {y}"
