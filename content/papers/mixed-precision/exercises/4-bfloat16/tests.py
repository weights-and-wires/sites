import warnings

import numpy as np
from submission import to_bfloat16

warnings.filterwarnings("ignore", category=RuntimeWarning)


def f(v):
    return float(to_bfloat16(np.array([v], dtype=np.float32))[0])


def test_dtype_and_shape():
    out = to_bfloat16(np.zeros((2, 3), dtype=np.float32))
    assert out.dtype == np.float32 and out.shape == (2, 3), f"expected float32 (2, 3), got {out.dtype} {out.shape}"


def test_representable_values_are_unchanged():
    for v in [1.0, 1.0078125, -2.5, 0.0, 65536.0]:
        assert f(v) == v, f"{v} is a bfloat16 value; expected {v}, got {f(v)}"


def test_round_to_nearest():
    assert f(3.14159265) == 3.140625, f"expected 3.140625, got {f(3.14159265)}"
    assert f(1.0 + 0.7 * 2.0**-7) == 1.0078125, f"expected 1.0078125, got {f(1.0 + 0.7 * 2.0**-7)}"
    assert f(1.0 + 0.3 * 2.0**-7) == 1.0, f"expected 1.0, got {f(1.0 + 0.3 * 2.0**-7)}"


def test_ties_go_to_even():
    assert f(1.0 + 2.0**-8) == 1.0, f"the tie 1 + 2^-8 should round to 1.0, got {f(1.0 + 2.0**-8)}"
    assert f(1.0 + 3 * 2.0**-8) == 1.015625, f"the tie 1 + 3*2^-8 should round to 1.015625, got {f(1.0 + 3 * 2.0**-8)}"


def test_negative_values_are_symmetric():
    for v in [3.14159265, 1.0 + 3 * 2.0**-8, 1e-30]:
        assert f(-v) == -f(v), f"to_bfloat16(-{v}) = {f(-v)} but -to_bfloat16({v}) = {-f(v)}"


def test_range_matches_fp32_not_fp16():
    assert f(1e-30) != 0.0, f"1e-30 is a normal FP32 number and must survive, got {f(1e-30)}"
    assert abs(f(1e-30) - 1e-30) / 1e-30 < 2.0**-7, f"relative error must be below 2^-7, got {f(1e-30)}"
    assert np.isfinite(f(1e38)), f"1e38 is within the FP32 range, got {f(1e38)}"
    assert float(np.float16(1e-30)) == 0.0, "FP16 loses 1e-30; this check guards the comparison"


def test_relative_error_bound():
    rng = np.random.default_rng(0)
    x = (rng.normal(size=1000) * 10.0 ** rng.integers(-20, 20, size=1000)).astype(np.float32)
    y = to_bfloat16(x)
    rel = np.abs(y - x) / np.abs(x)
    assert rel.max() <= 2.0**-8 * 1.0001, f"round to nearest keeps the relative error within 2^-8, got {rel.max():.3e}"
    u = y.view(np.uint32)
    assert np.all(u & np.uint32(0xFFFF) == 0), "the low 16 bits of every output must be zero"
