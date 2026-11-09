import numpy as np
from submission import alignment_matrix


def make(seed, Ty=4, Tx=6, n=3, n_prime=5):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(Ty, n)), rng.normal(size=(Tx, 2 * n)), rng.normal(size=(n_prime, n)),
            rng.normal(size=(n_prime, 2 * n)), rng.normal(size=n_prime))


def reference(S, H, Wa, Ua, va):
    """Equations 5 and 6 with explicit loops."""
    Ty, Tx = S.shape[0], H.shape[0]
    alpha = np.zeros((Ty, Tx))
    C = np.zeros((Ty, H.shape[1]))
    for i in range(Ty):
        e = np.array([va @ np.tanh(Wa @ S[i] + Ua @ H[j]) for j in range(Tx)])
        w = np.exp(e - e.max())
        alpha[i] = w / w.sum()
        C[i] = sum(alpha[i, j] * H[j] for j in range(Tx))
    return alpha, C


def test_shapes():
    alpha, C = alignment_matrix(*make(0))
    assert alpha.shape == (4, 6), f"alpha shape: expected (4, 6), got {alpha.shape}"
    assert C.shape == (4, 6), f"C shape: expected (4, 6), got {C.shape}"


def test_rows_sum_to_one():
    alpha, _ = alignment_matrix(*make(1))
    assert np.allclose(alpha.sum(axis=1), 1.0), f"each decoder step's weights must sum to 1 over the source, got {alpha.sum(axis=1)}"


def test_matches_loop_reference():
    args = make(2)
    alpha, C = alignment_matrix(*args)
    want_alpha, want_C = reference(*args)
    assert np.allclose(alpha, want_alpha), f"alpha differs from the loop reference\nexpected {want_alpha[0]}\ngot {alpha[0]}"
    assert np.allclose(C, want_C), f"C differs from the loop reference\nexpected {want_C[0]}\ngot {C[0]}"


def test_not_square_inputs():
    # Ty != Tx and n != 2n catch swapped axes.
    args = make(3, Ty=2, Tx=7, n=2, n_prime=4)
    alpha, C = alignment_matrix(*args)
    want_alpha, want_C = reference(*args)
    assert alpha.shape == (2, 7), f"expected (2, 7), got {alpha.shape}"
    assert np.allclose(alpha, want_alpha), "alpha differs from the loop reference for Ty=2, Tx=7"
    assert np.allclose(C, want_C), "C differs from the loop reference for Ty=2, Tx=7"


def test_each_step_uses_its_own_state():
    S, H, Wa, Ua, va = make(4)
    alpha, _ = alignment_matrix(S, H, Wa, Ua, va)
    rows_differ = not np.allclose(alpha[0], alpha[1], atol=1e-3)
    assert rows_differ, f"two different decoder states gave the same row {alpha[0]}; apply tanh before the dot with v_a"


def test_one_step_equals_the_full_matrix_row():
    S, H, Wa, Ua, va = make(5)
    alpha, C = alignment_matrix(S, H, Wa, Ua, va)
    a1, C1 = alignment_matrix(S[2:3], H, Wa, Ua, va)
    assert np.allclose(a1[0], alpha[2]), f"step 2 alone: expected {alpha[2]}, got {a1[0]}"
    assert np.allclose(C1[0], C[2]), f"step 2 context alone: expected {C[2]}, got {C1[0]}"


def test_permuting_source_permutes_columns():
    S, H, Wa, Ua, va = make(6)
    perm = np.array([2, 5, 0, 3, 1, 4])
    alpha, C = alignment_matrix(S, H, Wa, Ua, va)
    alpha2, C2 = alignment_matrix(S, H[perm], Wa, Ua, va)
    assert np.allclose(alpha2, alpha[:, perm]), f"columns must follow the annotations\nexpected {alpha[:, perm][0]}\ngot {alpha2[0]}"
    assert np.allclose(C, C2), "contexts must not change when the source order changes"
