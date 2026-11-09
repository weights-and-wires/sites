import numpy as np
from submission import init_lora, lora_forward

W0 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
B1 = np.array([[1.0], [2.0]])
A1 = np.array([[1.0, 0.0, -1.0]])
X1 = np.array([[1.0, 2.0, 3.0]])


def test_init_shapes_and_zero_b():
    A, B = init_lora(5, 7, 3)
    assert A.shape == (3, 7), f"A expected shape (3, 7), got {A.shape}"
    assert B.shape == (5, 3), f"B expected shape (5, 3), got {B.shape}"
    assert np.all(B == 0), f"B must start at zero, got max |B| = {np.abs(B).max()}"
    assert np.abs(A).max() > 0, "A must not be zero, or no gradient can reach B"


def test_init_scale_of_a():
    A, _ = init_lora(4, 400, 50, seed=1, sigma=0.5)
    assert abs(A.std() - 0.5) < 0.02, f"A should have standard deviation 0.5, got {A.std():.4f}"
    assert abs(A.mean()) < 0.02, f"A should have mean 0, got {A.mean():.4f}"


def test_init_is_seeded():
    A1_, _ = init_lora(4, 6, 2, seed=3)
    A2_, _ = init_lora(4, 6, 2, seed=3)
    A3_, _ = init_lora(4, 6, 2, seed=4)
    assert np.array_equal(A1_, A2_), "the same seed must give the same A"
    assert not np.array_equal(A1_, A3_), "different seeds must give different A"


def test_zero_b_gives_base_output():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 8))
    W = rng.normal(size=(5, 8))
    A, B = init_lora(5, 8, 3)
    got = lora_forward(X, W, A, B, alpha=3.0)
    assert np.allclose(got, X @ W.T), f"with B = 0 the layer must equal the base layer; max diff {np.abs(got - X @ W.T).max()}"


def test_worked_example_from_the_page():
    got = lora_forward(X1, W0, A1, B1, alpha=1.0)
    assert np.allclose(got, [[5.0, 1.0]]), f"expected [[5, 1]], got {got}"


def test_alpha_over_r_scaling():
    got = lora_forward(X1, W0, A1, B1, alpha=2.0)
    assert np.allclose(got, [[3.0, -3.0]]), f"alpha = 2, r = 1 should double the update: expected [[3, -3]], got {got}"
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 3))
    B = rng.normal(size=(2, 4))
    X = rng.normal(size=(5, 3))
    base = X @ W0.T
    a4 = lora_forward(X, W0, A, B, alpha=4.0) - base
    a8 = lora_forward(X, W0, A, B, alpha=8.0) - base
    assert np.allclose(a4, X @ (B @ A).T), f"alpha = r = 4 should give scale 1; max diff {np.abs(a4 - X @ (B @ A).T).max()}"
    assert np.allclose(a8, 2 * a4), "doubling alpha must double the update and leave the frozen path alone"


def test_matches_dense_delta_w():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(7, 6))
    W = rng.normal(size=(4, 6))
    A = rng.normal(size=(2, 6))
    B = rng.normal(size=(4, 2))
    want = X @ (W + (3.0 / 2) * B @ A).T
    got = lora_forward(X, W, A, B, alpha=3.0)
    assert got.shape == (7, 4), f"expected shape (7, 4), got {got.shape}"
    assert np.allclose(got, want), f"expected X (W0 + (alpha/r) B A)^T = {want[0]}, got {got[0]}"
