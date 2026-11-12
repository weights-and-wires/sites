import numpy as np
from submission import output_lattice

# The toy from the page: exp(f) and exp(g) are small integers.
F = np.array([[1, 3, 1], [1, 1, 3], [2, 1, 1]], dtype=float)
G = np.array([[1, 1, 1], [1, 1, 2], [4, 1, 1]], dtype=float)
Y = (1, 2)


def test_shapes():
    emit, null = output_lattice(np.log(F), np.log(G), Y)
    assert emit.shape == (3, 2), f"expected emit shape (3, 2), got {emit.shape}"
    assert null.shape == (3, 3), f"expected null shape (3, 3), got {null.shape}"


def test_node_from_the_page():
    # At (t=2, u=1): h = exp(f_2) * exp(g_1) = [1, 1, 6], which sums to 8.
    emit, null = output_lattice(np.log(F), np.log(G), Y)
    assert abs(null[1, 1] - 1 / 8) < 1e-12, f"expected null(2, 1) = 0.125, got {null[1, 1]}"
    assert abs(emit[1, 1] - 6 / 8) < 1e-12, f"expected y(2, 1) = P(b | 2, 1) = 0.75, got {emit[1, 1]}"


def test_full_table_from_the_page():
    emit, null = output_lattice(np.log(F), np.log(G), Y)
    want_null = np.array([[0.2, 1 / 6, 0.5], [0.2, 0.125, 0.5], [0.5, 0.4, 0.8]])
    want_emit = np.array([[0.6, 1 / 3], [0.2, 0.75], [0.25, 0.4]])
    assert np.allclose(null, want_null), f"expected null\n{want_null}\ngot\n{null}"
    assert np.allclose(emit, want_emit), f"expected emit\n{want_emit}\ngot\n{emit}"


def test_label_follows_the_target_position():
    # Swapping the target must swap which column emit reads.
    emit_ab, _ = output_lattice(np.log(F), np.log(G), (1, 2))
    emit_ba, _ = output_lattice(np.log(F), np.log(G), (2, 1))
    assert abs(emit_ab[0, 0] - 0.6) < 1e-12, f"for y = (a, b) expected P(a | 1, 0) = 0.6, got {emit_ab[0, 0]}"
    assert abs(emit_ba[0, 0] - 0.2) < 1e-12, f"for y = (b, a) expected P(b | 1, 0) = 0.2, got {emit_ba[0, 0]}"
    assert abs(emit_ab[0, 1] - 1 / 3) < 1e-12, f"for y = (a, b) expected P(b | 1, 1) = 1/3, got {emit_ab[0, 1]}"
    assert abs(emit_ba[0, 1] - 0.5) < 1e-12, f"for y = (b, a) expected P(a | 1, 1) = 0.5, got {emit_ba[0, 1]}"


def test_sums_over_symbols_use_all_columns():
    rng = np.random.default_rng(3)
    f = rng.normal(size=(4, 5))
    g = rng.normal(size=(3, 5))
    emit, null = output_lattice(f, g, (2, 4))
    logits = f[:, None, :] + g[None, :, :]
    p = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    assert np.allclose(null, p[:, :, 0]), "null is not the softmax over the symbol axis at column 0"
    assert np.allclose(emit[:, 0], p[:, 0, 2]), f"emit[:, 0] should read column 2, got {emit[:, 0]}"
    assert np.allclose(emit[:, 1], p[:, 1, 4]), f"emit[:, 1] should read column 4, got {emit[:, 1]}"
    assert np.all(null + emit[:, :1] <= 1 + 1e-12), "null plus the emit probability can't exceed 1"


def test_large_logits_do_not_overflow():
    f = np.log(F) + 1000.0
    emit, null = output_lattice(f, np.log(G), Y)
    assert np.all(np.isfinite(emit)) and np.all(np.isfinite(null)), f"expected finite values, got {emit}, {null}"
    assert abs(null[1, 1] - 0.125) < 1e-9, f"a constant shift must not change the softmax; expected 0.125, got {null[1, 1]}"


def test_empty_target():
    emit, null = output_lattice(np.log(F), np.log(G)[:1], ())
    assert emit.shape == (3, 0), f"expected emit shape (3, 0), got {emit.shape}"
    assert null.shape == (3, 1), f"expected null shape (3, 1), got {null.shape}"
    assert np.allclose(null[:, 0], [1 / 5, 1 / 5, 1 / 2]), f"expected [0.2, 0.2, 0.5], got {null[:, 0]}"
