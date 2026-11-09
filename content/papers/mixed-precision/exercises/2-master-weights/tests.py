import numpy as np
from submission import fp16_only_step, master_step


def run(steps, w0, g, lr):
    w16 = np.array([w0], dtype=np.float16)
    master = np.array([w0], dtype=np.float32)
    g16 = np.array([g], dtype=np.float16)
    for _ in range(steps):
        w16 = fp16_only_step(w16, g16, lr)
    m = master.copy()
    for _ in range(steps):
        m, w_copy = master_step(m, g16, lr)
    return w16, m, w_copy


def test_dtypes():
    m, w = master_step(np.array([1.0], dtype=np.float32), np.array([0.1], dtype=np.float16), 0.1)
    assert m.dtype == np.float32, f"master must stay float32, got {m.dtype}"
    assert w.dtype == np.float16, f"the working copy must be float16, got {w.dtype}"
    out = fp16_only_step(np.array([1.0], dtype=np.float16), np.array([0.1], dtype=np.float16), 0.1)
    assert out.dtype == np.float16, f"fp16_only_step must return float16, got {out.dtype}"


def test_one_step_matches_sgd():
    m, w = master_step(np.array([1.0, 2.0], dtype=np.float32), np.array([0.5, -0.5], dtype=np.float16), 0.1)
    assert np.allclose(m, [0.95, 2.05], atol=1e-6), f"expected [0.95, 2.05], got {m}"
    assert np.allclose(w, [0.95, 2.05], atol=1e-3), f"expected about [0.95, 2.05], got {w}"


def test_large_weight_small_update_is_lost_in_fp16_only():
    # lr * g = 2e-4 is representable in FP16, but 1.0 - 2e-4 rounds back to 1.0.
    w16, m, w_copy = run(1000, 1.0, 0.2, 1e-3)
    assert float(w16[0]) == 1.0, f"the FP16-only weight must stall at 1.0, got {float(w16[0])}"
    assert abs(float(m[0]) - 0.8) < 0.005, f"the master should reach about 0.8, got {float(m[0])}"
    assert abs(float(w_copy[0]) - 0.8) < 0.005, f"the FP16 copy of the master should be about 0.8, got {float(w_copy[0])}"


def test_update_that_underflows_in_fp16_survives_in_master():
    # lr * g = 1e-8 is below 2^-25, so it is 0 in FP16.
    w16, m, w_copy = run(1000, 0.0, 1e-4, 1e-4)
    assert float(w16[0]) == 0.0, f"the FP16-only weight must not move, got {float(w16[0])}"
    assert abs(float(m[0]) + 1e-5) < 2e-7, f"the master should move to about -1e-5, got {float(m[0])}"


def test_update_is_computed_in_fp32():
    # lr and g are each exact in FP16, but lr * g = 2^-30 only exists in FP32.
    m, _ = master_step(np.array([0.0], dtype=np.float32), np.array([2.0**-15], dtype=np.float16), 2.0**-15)
    assert m[0] == -(2.0**-30), f"expected {-(2.0**-30)}, got {m[0]}"


def test_working_copy_is_rounded_from_master_not_accumulated():
    m = np.array([1.0], dtype=np.float32)
    g16 = np.array([0.2], dtype=np.float16)
    for _ in range(1000):
        m, w = master_step(m, g16, 1e-3)
    assert w[0] == np.float16(m[0]), f"expected the FP16 rounding of the master {np.float16(m[0])}, got {w[0]}"
