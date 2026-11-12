import math

import numpy as np
from submission import bias_estimates

N = 100000


def test_keys_and_types():
    out = bias_estimates(2, 1.0, 1000, 0)
    assert set(out) == {"max", "double", "clipped"}, f"expected keys max, double, clipped; got {sorted(out)}"
    assert all(isinstance(v, float) for v in out.values()), f"values must be floats, got {[type(v) for v in out.values()]}"


def test_max_is_biased_upward_two_actions():
    out = bias_estimates(2, 1.0, N, 0)
    want = 1 / math.sqrt(math.pi)
    assert abs(out["max"] - want) < 0.02, f"expected about {want:.4f}, got {out['max']:.4f}"


def test_max_bias_grows_with_the_number_of_actions():
    b2 = bias_estimates(2, 1.0, N, 1)["max"]
    b10 = bias_estimates(10, 1.0, N, 1)["max"]
    assert abs(b10 - 1.5388) < 0.03, f"expected about 1.5388 for 10 actions, got {b10:.4f}"
    assert b10 > b2 + 0.5, f"more actions must mean more bias: {b2:.4f} then {b10:.4f}"


def test_bias_scales_with_sigma():
    b1 = bias_estimates(10, 1.0, N, 2)["max"]
    b2 = bias_estimates(10, 2.0, N, 2)["max"]
    assert abs(b2 / b1 - 2.0) < 0.05, f"doubling sigma must double the bias: {b1:.4f}, {b2:.4f}"


def test_double_estimate_is_unbiased():
    out = bias_estimates(10, 1.0, N, 3)
    assert abs(out["double"]) < 0.02, f"the independent critic must give bias near 0, got {out['double']:.4f}"


def test_clipped_is_at_or_below_double_and_below_max():
    out = bias_estimates(2, 1.0, N, 4)
    assert out["clipped"] < out["double"] - 0.1, f"min must lower the estimate: clipped {out['clipped']:.4f}, double {out['double']:.4f}"
    assert out["clipped"] < 0, f"with 2 actions the clipped target underestimates, got {out['clipped']:.4f}"


def test_clipped_two_actions_value():
    out = bias_estimates(2, 1.0, N, 5)
    assert abs(out["clipped"] - (-0.2833)) < 0.02, f"expected about -0.283, got {out['clipped']:.4f}"


def test_seed_is_reproducible():
    a = bias_estimates(5, 1.0, 5000, 11)
    b = bias_estimates(5, 1.0, 5000, 11)
    assert a == b, f"the same seed must give the same numbers: {a} vs {b}"
