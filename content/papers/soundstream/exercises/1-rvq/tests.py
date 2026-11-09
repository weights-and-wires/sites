import numpy as np
from submission import rvq_encode, rvq_decode

C1 = np.array([[0, 0], [4, 0], [0, 4], [4, 4]], dtype=float)
C2 = np.array([[0, 0], [-1, 1], [1, -1], [1, 1]], dtype=float)
C3 = 0.25 * C2
BOOKS = [C1, C2, C3]
Y = np.array([[3.2, 0.9]])


def test_page_example_indices_and_value():
    idx, y_hat = rvq_encode(Y, BOOKS)
    assert idx.shape == (1, 3), f"expected indices of shape (1, 3), got {idx.shape}"
    assert idx.tolist() == [[1, 1, 2]], f"expected [[1, 1, 2]], got {idx.tolist()}"
    assert np.allclose(y_hat, [[3.25, 0.75]]), f"expected [[3.25, 0.75]], got {y_hat}"


def test_stage_two_quantizes_the_residual():
    # Quantizing y itself at stage 2 would pick [1, 1] (index 3) and give [5, 1].
    idx, y_hat = rvq_encode(Y, BOOKS[:2])
    assert idx[0, 1] == 1, f"stage 2 must pick the code nearest to the residual [-0.8, 0.9], index 1; got {idx[0, 1]}"
    assert np.allclose(y_hat, [[3.0, 1.0]]), f"expected [[3, 1]] after two stages, got {y_hat}"


def test_nearest_is_by_distance_not_dot_product():
    book = np.array([[0.0, 0.0], [4.0, 0.0]])
    idx, y_hat = rvq_encode(np.array([[1.5, 0.0]]), [book])
    assert idx.tolist() == [[0]], f"[1.5, 0] is nearest to [0, 0] (index 0); got {idx.tolist()}"
    assert np.allclose(y_hat, [[0.0, 0.0]]), f"expected [[0, 0]], got {y_hat}"


def test_batch_and_dtype():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 4))
    books = [rng.normal(size=(8, 4)) for _ in range(3)]
    idx, y_hat = rvq_encode(y, books)
    assert idx.shape == (50, 3) and y_hat.shape == (50, 4), f"expected shapes (50, 3) and (50, 4), got {idx.shape} and {y_hat.shape}"
    assert np.issubdtype(idx.dtype, np.integer), f"indices must be integers, got dtype {idx.dtype}"
    assert idx.min() >= 0 and idx.max() < 8, f"indices must be in [0, 8), got range [{idx.min()}, {idx.max()}]"


def test_error_never_grows_with_a_zero_code():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(200, 3)) * 3
    books = []
    for _ in range(4):
        b = rng.normal(size=(6, 3))
        b[0] = 0.0
        books.append(b)
    errs = [float((y ** 2).mean())]
    for n in range(1, 5):
        _, y_hat = rvq_encode(y, books[:n])
        errs.append(float(((y - y_hat) ** 2).mean()))
    for n in range(1, 5):
        assert errs[n] <= errs[n - 1] + 1e-12, f"error must not grow when a stage is added (a zero code is available); stage {n}: {errs[n - 1]:.4f} -> {errs[n]:.4f}"
    assert errs[4] < errs[1], f"four stages must beat one stage here, got {errs[4]:.4f} vs {errs[1]:.4f}"


def test_decode_matches_encode():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(30, 5))
    books = [rng.normal(size=(16, 5)) for _ in range(4)]
    idx, y_hat = rvq_encode(y, books)
    got = rvq_decode(idx, books)
    assert np.allclose(got, y_hat), f"decode must equal the encoder's y_hat; max difference {np.abs(got - y_hat).max()}"


def test_decode_with_fewer_stages():
    idx = np.array([[1, 1, 2]])
    got2 = rvq_decode(idx[:, :2], BOOKS)
    got1 = rvq_decode(idx[:, :1], BOOKS)
    assert np.allclose(got2, [[3.0, 1.0]]), f"two stages: expected [[3, 1]], got {got2}"
    assert np.allclose(got1, [[4.0, 0.0]]), f"one stage: expected [[4, 0]], got {got1}"
