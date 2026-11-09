import numpy as np
from submission import ar_mask, nar_mask, masked_softmax


def test_shape_and_dtype():
    m = ar_mask(3, 4)
    assert isinstance(m, np.ndarray) and m.dtype == bool, f"expected a boolean array, got {getattr(m, 'dtype', type(m))}"
    assert m.shape == (7, 7), f"expected shape (7, 7), got {m.shape}"
    assert nar_mask(3, 4).shape == (7, 7), f"expected shape (7, 7), got {nar_mask(3, 4).shape}"


def test_audio_rows_follow_the_paper():
    # Section 4.2.1: c_{t,1} attends to (x, c_{<=t,1}).
    n_text, n_audio = 3, 4
    m = ar_mask(n_text, n_audio)
    for t in range(n_audio):
        want = np.array([True] * n_text + [j <= t for j in range(n_audio)])
        got = m[n_text + t]
        assert np.array_equal(got, want), f"audio row {t}: expected {want.astype(int)}, got {got.astype(int)}"


def test_text_rows_never_see_audio():
    m = ar_mask(3, 4)
    assert not m[:3, 3:].any(), f"text rows must not attend to acoustic tokens, got\n{m[:3, 3:].astype(int)}"


def test_every_row_has_a_key():
    m = ar_mask(2, 2)
    assert m.any(axis=1).all(), f"every query needs at least one key, got row sums {m.sum(axis=1)}"


def test_ar_weights_ignore_the_future():
    rng = np.random.default_rng(0)
    n_text, n_audio = 2, 5
    n = n_text + n_audio
    w = masked_softmax(rng.normal(size=(n, n)), ar_mask(n_text, n_audio))
    future = np.triu(w[n_text:, n_text:], k=1)
    assert np.allclose(future, 0, atol=1e-8), f"future acoustic weights must be 0, got max {future.max()}"
    assert np.allclose(w.sum(-1), 1), f"rows must sum to 1, got {w.sum(-1)}"
    assert (w[n_text:, :n_text] > 0).all(), "acoustic rows must put weight on the text"


def test_nar_sees_everything():
    m = nar_mask(2, 3)
    assert m.dtype == bool and m.all(), f"expected an all-True 5 x 5 mask, got\n{m.astype(int)}"
    assert m.shape == (5, 5), f"expected shape (5, 5), got {m.shape}"


def test_degenerate_sizes():
    m = ar_mask(1, 1)
    assert np.array_equal(m[1], [True, True]), f"expected [True, True] for the audio row, got {m[1]}"
    assert not m[0, 1], "the text token must not see the audio token"
