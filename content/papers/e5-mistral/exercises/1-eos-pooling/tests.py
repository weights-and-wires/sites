import numpy as np
from submission import last_token_pool, embed


def toy_hidden():
    # hidden[b, t] = (10 * b + t, -t): every position holds a unique vector.
    h = np.zeros((2, 4, 2))
    for b in range(2):
        for t in range(4):
            h[b, t] = (10 * b + t, -t)
    return h


def test_right_padding():
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    got = last_token_pool(toy_hidden(), mask)
    want = np.array([[3.0, -3.0], [11.0, -1.0]])
    assert got.shape == (2, 2), f"expected shape (2, 2), got {got.shape}"
    assert np.allclose(got, want), f"right padding: expected {want.tolist()}, got {got.tolist()}"


def test_left_padding():
    mask = np.array([[1, 1, 1, 1], [0, 0, 1, 1]])
    got = last_token_pool(toy_hidden(), mask)
    want = np.array([[3.0, -3.0], [13.0, -3.0]])
    assert np.allclose(got, want), f"left padding: expected {want.tolist()}, got {got.tolist()}"


def test_mixed_padding_sides_and_lengths():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(4, 6, 3))
    mask = np.array([
        [1, 1, 1, 0, 0, 0],   # right padded, last real index 2
        [0, 0, 0, 0, 1, 1],   # left padded, last real index 5
        [1, 1, 1, 1, 1, 1],   # no padding
        [1, 0, 0, 0, 0, 0],   # a single real token at index 0
    ])
    want = h[np.arange(4), [2, 5, 5, 0]]
    got = last_token_pool(h, mask)
    assert np.allclose(got, want), f"expected rows at indices [2, 5, 5, 0], got\n{got}\nwant\n{want}"


def test_padding_values_are_ignored():
    h = toy_hidden()
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    h2 = h.copy()
    h2[1, 2:] = 1e6   # garbage at the padded positions
    a, b = last_token_pool(h, mask), last_token_pool(h2, mask)
    assert np.allclose(a, b), f"padded positions must not matter; got {a.tolist()} and {b.tolist()}"


def test_embed_is_unit_length_and_same_direction():
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    out = embed(toy_hidden(), mask)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0), f"expected unit norms, got {norms}"
    want = np.array([3.0, -3.0]) / np.sqrt(18.0)
    assert np.allclose(out[0], want), f"row 0: expected {want}, got {out[0]}"
    want1 = np.array([11.0, -1.0]) / np.sqrt(122.0)
    assert np.allclose(out[1], want1), f"row 1: expected {want1}, got {out[1]}"
