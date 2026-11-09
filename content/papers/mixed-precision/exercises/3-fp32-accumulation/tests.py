import warnings

import numpy as np
from submission import dot

warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_small_dot_is_exact():
    a = np.array([1, 2, 3], dtype=np.float16)
    b = np.array([4, 5, 6], dtype=np.float16)
    for acc in (np.float16, np.float32):
        got = dot(a, b, acc)
        assert float(got) == 32.0, f"{acc.__name__}: expected 32.0, got {float(got)}"


def test_returns_float16_scalar():
    a = np.ones(4, dtype=np.float16)
    for acc in (np.float16, np.float32):
        got = dot(a, a, acc)
        assert np.asarray(got).dtype == np.float16, f"{acc.__name__}: expected float16, got {np.asarray(got).dtype}"


def test_fp32_accumulator_counts_to_4096():
    a = np.ones(4096, dtype=np.float16)
    got = float(dot(a, a, np.float32))
    assert got == 4096.0, f"expected 4096.0, got {got}"


def test_fp16_accumulator_stalls_at_2048():
    a = np.ones(4096, dtype=np.float16)
    got = float(dot(a, a, np.float16))
    assert got == 2048.0, f"expected the FP16 accumulator to stall at 2048.0, got {got}"


def test_random_vectors():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 4096).astype(np.float16)
    b = rng.uniform(0, 1, 4096).astype(np.float16)
    exact = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    f32 = float(dot(a, b, np.float32))
    f16 = float(dot(a, b, np.float16))
    assert abs(f32 - exact) / exact < 1e-3, f"FP32 accumulation should be within 0.1% of {exact:.2f}, got {f32:.2f}"
    assert abs(f16 - exact) / exact > 0.02, f"FP16 accumulation should lose more than 2% of {exact:.2f}, got {f16:.2f}"


def test_products_are_formed_in_the_accumulator_type():
    # 300 * 300 = 90000 overflows FP16 but not FP32; 0.1 rounded in FP16 differs from FP32.
    a = np.array([300.0], dtype=np.float16)
    assert np.isinf(float(dot(a, a, np.float16))), "an FP16 product of 300 * 300 must overflow to inf"
    got = float(dot(np.array([300.0, 300.0], dtype=np.float16), np.array([300.0, -300.0], dtype=np.float16), np.float32))
    assert got == 0.0, f"the FP32 accumulator holds 90000 and cancels to 0.0, got {got}"
