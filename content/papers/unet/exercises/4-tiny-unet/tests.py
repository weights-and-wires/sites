import numpy as np
from submission import tiny_unet, upconv2x2


def make_params(K=2, seed=0):
    r = np.random.default_rng(seed)

    def layer(o, i, k):
        return (r.normal(scale=0.4, size=(o, i, k, k)), r.normal(scale=0.1, size=o))

    return {
        "c1": layer(2, 1, 3),
        "c2": layer(2, 2, 3),
        "c3": layer(4, 2, 3),
        "c4": layer(4, 4, 3),
        "up": (r.normal(scale=0.4, size=(4, 2, 2, 2)), r.normal(scale=0.1, size=2)),
        "c5": layer(2, 4, 3),
        "c6": layer(2, 2, 3),
        "head": layer(K, 2, 1),
    }


def test_upconv_delta_paints_the_kernel():
    x = np.zeros((1, 2, 2))
    x[0, 0, 0] = 1.0
    W = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # (Cin=1, Cout=1, 2, 2)
    out = upconv2x2(x, W, np.zeros(1))
    want = np.zeros((1, 4, 4))
    want[0, :2, :2] = [[1, 2], [3, 4]]
    assert np.array_equal(out, want), f"expected the kernel in the top-left 2x2 block\n{want[0]}\ngot\n{out[0]}"


def test_upconv_blocks_do_not_overlap():
    x = np.array([[[1.0, 10.0], [100.0, 1000.0]]])
    W = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    out = upconv2x2(x, W, np.zeros(1))
    assert out.shape == (1, 4, 4), f"expected shape (1, 4, 4), got {out.shape}"
    assert np.array_equal(out[0, 2:, 2:], 1000 * W[0, 0]), f"bottom-right block must be 1000 * kernel, got\n{out[0, 2:, 2:]}"
    assert np.array_equal(out[0, :2, 2:], 10 * W[0, 0]), f"top-right block must be 10 * kernel, got\n{out[0, :2, 2:]}"


def test_upconv_channels_and_bias():
    r = np.random.default_rng(3)
    x = r.normal(size=(3, 2, 2))
    W = r.normal(size=(3, 2, 2, 2))
    b = np.array([0.5, -1.0])
    out = upconv2x2(x, W, b)
    want = sum(x[i, 1, 0] * W[i, 1, 0, 1] for i in range(3)) + b[1]
    assert out.shape == (2, 4, 4), f"expected shape (2, 4, 4), got {out.shape}"
    assert np.isclose(out[1, 2, 1], want), f"out[1, 2, 1] expected {want}, got {out[1, 2, 1]}"


def test_output_shape_20():
    out = tiny_unet(np.random.default_rng(1).normal(size=(1, 20, 20)), make_params())
    assert out.shape == (2, 4, 4), f"expected (2, 4, 4) for a 1x20x20 input, got {out.shape}"


def test_output_shape_36():
    out = tiny_unet(np.random.default_rng(1).normal(size=(1, 36, 36)), make_params())
    assert out.shape == (2, 20, 20), f"expected (2, 20, 20) for a 1x36x36 input, got {out.shape}"


def test_golden_values():
    out = tiny_unet(np.random.default_rng(1).normal(size=(1, 20, 20)), make_params())
    assert np.isclose(out.sum(), 19.985209630074944, atol=1e-6), f"expected sum 19.9852, got {out.sum()}"
    assert np.isclose(out[0, 0, 0], 0.2691387579167457, atol=1e-6), f"out[0, 0, 0] expected 0.2691, got {out[0, 0, 0]}"
    assert np.isclose(out[1, 3, 3], -1.820333263485508, atol=1e-6), f"out[1, 3, 3] expected -1.8203, got {out[1, 3, 3]}"


def test_skip_path_carries_the_signal():
    # With a zero up-convolution, only the skip channels reach the expanding convolutions.
    P = make_params()
    P["up"] = (np.zeros((4, 2, 2, 2)), np.zeros(2))
    out = tiny_unet(np.random.default_rng(1).normal(size=(1, 20, 20)), P)
    assert np.isclose(out.sum(), -2.917263186609513, atol=1e-6), f"expected sum -2.9173 with the skip only, got {out.sum()}"
    assert np.isclose(out[0, 0, 0], -0.1902708716496091, atol=1e-6), f"out[0, 0, 0] expected -0.1903, got {out[0, 0, 0]}"

