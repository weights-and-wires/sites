import numpy as np
from submission import layer_stds

N, H, L = 32, 128, 20


def make(scale, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, H))
    ws = [rng.normal(size=(H, H)) * scale / np.sqrt(H) for _ in range(L)]
    return x, ws


def test_shape():
    x, ws = make(1.0)
    s = layer_stds(x, ws, use_ln=True)
    assert np.asarray(s).shape == (L, N), f"expected shape {(L, N)}, got {np.asarray(s).shape}"


def test_layer_norm_pins_every_example_at_every_layer():
    for scale in (2.0, 1.0):
        x, ws = make(scale)
        s = layer_stds(x, ws, use_ln=True)
        assert np.allclose(s, 1.0, atol=1e-3), f"scale {scale}: expected std 1 everywhere, got min {s.min():.4f} and max {s.max():.4f}"


def test_first_layer_without_ln_matches_the_weight_scale():
    x, ws = make(2.0)
    s = layer_stds(x, ws, use_ln=False)
    assert abs(s[0].mean() - 2.0) < 0.2, f"layer 1 std expected about 2.0, got {s[0].mean():.3f}"


def test_signal_explodes_without_ln_at_scale_2():
    x, ws = make(2.0)
    s = layer_stds(x, ws, use_ln=False)
    assert s[-1].mean() > 100, f"expected the layer-20 std to exceed 100, got {s[-1].mean():.3g}"
    assert s[-1].mean() > 10 * s[4].mean(), f"std should keep growing: layer 5 is {s[4].mean():.3g}, layer 20 is {s[-1].mean():.3g}"


def test_signal_vanishes_without_ln_at_scale_1():
    x, ws = make(1.0)
    s = layer_stds(x, ws, use_ln=False)
    assert s[-1].mean() < 0.01, f"expected the layer-20 std below 0.01, got {s[-1].mean():.3g}"


def test_layers_are_applied_in_order():
    x, ws = make(1.0)
    s = layer_stds(x, ws, use_ln=False)
    # Without normalization, std at layer 2 depends on the layer-1 output; recompute by hand.
    h = np.maximum(x @ ws[0].T, 0)
    want = (h @ ws[1].T).std(axis=-1)
    assert np.allclose(s[1], want), f"layer 2 expected {want[:3]}..., got {s[1][:3]}..."


def test_ln_output_feeds_the_next_layer():
    x, ws = make(2.0)
    s = layer_stds(x, ws, use_ln=True)
    a1 = x @ ws[0].T
    mu, var = a1.mean(-1, keepdims=True), a1.var(-1, keepdims=True)
    h = np.maximum((a1 - mu) / np.sqrt(var + 1e-5), 0)
    a2 = h @ ws[1].T
    mu, var = a2.mean(-1, keepdims=True), a2.var(-1, keepdims=True)
    want = ((a2 - mu) / np.sqrt(var + 1e-5)).std(axis=-1)
    assert np.allclose(s[1], want), f"layer 2 expected {want[:3]}..., got {s[1][:3]}..."
