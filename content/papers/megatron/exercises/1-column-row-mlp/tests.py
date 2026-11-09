import numpy as np
from submission import split_columns, split_rows, tp_mlp, gelu

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 8))
A = rng.normal(size=(8, 32))
B = rng.normal(size=(32, 8))
FULL = gelu(X @ A) @ B


def test_split_columns():
    M = np.arange(12.0).reshape(3, 4)
    parts = split_columns(M, 2)
    assert len(parts) == 2, f"expected 2 blocks, got {len(parts)}"
    assert np.array_equal(parts[0], M[:, :2]) and np.array_equal(parts[1], M[:, 2:]), f"expected column blocks of {M.tolist()}, got {[p.tolist() for p in parts]}"


def test_split_rows():
    M = np.arange(12.0).reshape(4, 3)
    parts = split_rows(M, 2)
    assert len(parts) == 2, f"expected 2 blocks, got {len(parts)}"
    assert np.array_equal(parts[0], M[:2]) and np.array_equal(parts[1], M[2:]), f"expected row blocks of {M.tolist()}, got {[p.tolist() for p in parts]}"


def test_matches_single_device():
    for n in (1, 2, 4, 8):
        got = tp_mlp(X, A, B, n)
        assert np.allclose(got, FULL), f"n={n}: expected max |diff| < 1e-8, got {np.abs(got - FULL).max()}"


def test_exactly_one_reduction_of_full_size_partials():
    calls = []

    def counting(parts):
        calls.append([p.shape for p in parts])
        return sum(parts)

    tp_mlp(X, A, B, 4, reduce_fn=counting)
    assert len(calls) == 1, f"expected exactly 1 all-reduce, got {len(calls)}"
    assert calls[0] == [(5, 8)] * 4, f"expected 4 partial outputs of shape (5, 8), got {calls[0]}"


def test_a_single_partial_is_not_the_answer():
    seen = []

    def spy(parts):
        seen.extend(parts)
        return sum(parts)

    tp_mlp(X, A, B, 4, reduce_fn=spy)
    assert not np.allclose(seen[0], FULL), "one GPU's partial output must differ from Z; the reduction is what completes it"


def test_gelu_is_applied_before_the_reduction():
    # With one big negative and one big positive pre-activation, gelu(sum) and sum(gelu) differ a lot.
    X1 = np.array([[1.0, 2.0]])
    A1 = np.array([[3.0, 3.0], [-2.0, -2.0]])
    B1 = np.eye(2)
    want = gelu(X1 @ A1[:, :1]) @ B1[:1] + gelu(X1 @ A1[:, 1:]) @ B1[1:]
    got = tp_mlp(X1, A1, B1, 2)
    assert np.allclose(got, want), f"expected {want}, got {got}"
    assert np.allclose(got, gelu(X1 @ A1) @ B1), f"expected {gelu(X1 @ A1) @ B1}, got {got}"
