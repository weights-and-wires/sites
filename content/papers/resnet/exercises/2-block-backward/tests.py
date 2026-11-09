import numpy as np
from submission import block_backward, block_forward


def setup(seed=0, d=5, h=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=d), rng.normal(size=(h, d)), rng.normal(size=(d, h)), rng.normal(size=d)


def numeric(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps
        hi = f()
        arr[i] = old - eps
        lo = f()
        arr[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


def test_input_gradient_matches_finite_differences():
    x, W1, W2, c = setup()
    g_x, _, _ = block_backward(x, W1, W2, c)
    want = numeric(lambda: float(c @ block_forward(x, W1, W2)), x)
    assert np.allclose(g_x, want, atol=1e-5), f"expected dL/dx = {want}, got {g_x}"


def test_weight_gradients_match_finite_differences():
    x, W1, W2, c = setup(1)
    _, gW1, gW2 = block_backward(x, W1, W2, c)
    want1 = numeric(lambda: float(c @ block_forward(x, W1, W2)), W1)
    want2 = numeric(lambda: float(c @ block_forward(x, W1, W2)), W2)
    assert gW1.shape == W1.shape and gW2.shape == W2.shape, f"expected shapes {W1.shape} and {W2.shape}, got {gW1.shape} and {gW2.shape}"
    assert np.allclose(gW1, want1, atol=1e-5), f"expected dL/dW1 =\n{want1}\ngot\n{gW1}"
    assert np.allclose(gW2, want2, atol=1e-5), f"expected dL/dW2 =\n{want2}\ngot\n{gW2}"


def test_identity_path_survives_zero_weights():
    x = np.array([1.0, 2.0, 3.0])
    Z1, Z2 = np.zeros((4, 3)), np.zeros((3, 4))
    g = np.array([0.5, -1.0, 2.0])
    g_x, gW1, gW2 = block_backward(x, Z1, Z2, g)
    assert np.allclose(g_x, g), f"with F = 0 the gradient passes through unchanged; expected {g}, got {g_x}"
    assert np.allclose(gW1, 0), f"expected dL/dW1 = 0 when W2 = 0, got\n{gW1}"


def test_output_relu_gates_the_gradient():
    x = np.array([-2.0, 3.0])
    Z1, Z2 = np.zeros((2, 2)), np.zeros((2, 2))
    g = np.array([1.0, 1.0])
    g_x, _, _ = block_backward(x, Z1, Z2, g)
    want = np.array([0.0, 1.0])
    assert np.allclose(g_x, want), f"z = [-2, 3], so the first entry is cut by the output ReLU; expected {want}, got {g_x}"


def test_gradient_through_f_adds_to_shortcut():
    x = np.array([1.0])
    W1, W2 = np.array([[2.0]]), np.array([[3.0]])
    g_x, _, _ = block_backward(x, W1, W2, np.array([1.0]))
    # y = 3 * relu(2x) + x = 7x for x > 0, so dy/dx = 7 (6 through F, 1 through the shortcut)
    assert np.allclose(g_x, [7.0]), f"expected dy/dx = 7 = 6 + 1, got {g_x}"
