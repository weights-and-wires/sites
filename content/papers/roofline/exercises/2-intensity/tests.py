import math
from submission import layernorm_intensity, matmul_intensity, softmax_intensity


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9)


def test_matmul_square_fp32():
    f, t, i = matmul_intensity(512, 512, 512, 4)
    assert f == 2 * 512 ** 3, f"expected {2 * 512 ** 3} FLOPs, got {f}"
    assert t == 4 * 3 * 512 ** 2, f"expected {4 * 3 * 512 ** 2} bytes, got {t}"
    assert close(i, 512 / 6), f"expected intensity {512 / 6:.4f}, got {i}"


def test_matmul_rectangular():
    f, t, i = matmul_intensity(2, 3, 5, 4)
    assert f == 60, f"expected 60 FLOPs for m=2,k=3,n=5, got {f}"
    assert t == 4 * (6 + 15 + 10), f"expected {4 * 31} bytes (A 6, B 15, C 10 elements), got {t}"
    assert close(i, 60 / 124), f"expected intensity {60 / 124:.5f}, got {i}"


def test_matmul_intensity_grows_with_size():
    small = matmul_intensity(64, 64, 64, 4)[2]
    big = matmul_intensity(4096, 4096, 4096, 4)[2]
    assert big > 50 * small, f"intensity should grow about linearly with n; got {small} and {big}"


def test_softmax():
    f, t, i = softmax_intensity(4096, 4096, 4)
    assert f == 5 * 4096 * 4096, f"expected {5 * 4096 * 4096} FLOPs, got {f}"
    assert t == 8 * 4096 * 4096, f"expected {8 * 4096 * 4096} bytes (read plus write), got {t}"
    assert close(i, 0.625), f"expected FP32 intensity 0.625, got {i}"


def test_bytes_per_element_scales_intensity():
    i32 = softmax_intensity(128, 128, 4)[2]
    i16 = softmax_intensity(128, 128, 2)[2]
    assert close(i16, 2 * i32), f"halving the element size should double the intensity: {i32} vs {i16}"


def test_layernorm_counts_the_vectors():
    f, t, i = layernorm_intensity(4, 8, 4)
    assert f == 8 * 4 * 8, f"expected {8 * 4 * 8} FLOPs, got {f}"
    assert t == 4 * (2 * 4 * 8 + 2 * 8), f"expected {4 * (2 * 4 * 8 + 2 * 8)} bytes including gamma and beta, got {t}"
    assert close(i, 256 / 320), f"expected intensity {256 / 320}, got {i}"


def test_layernorm_large():
    i = layernorm_intensity(4096, 4096, 4)[2]
    assert abs(i - 1.0) < 1e-3, f"expected about 1.0 FLOPs/byte for a large FP32 matrix, got {i}"


def test_ordering():
    mm = matmul_intensity(4096, 4096, 4096, 2)[2]
    sm = softmax_intensity(4096, 4096, 2)[2]
    ln = layernorm_intensity(4096, 4096, 2)[2]
    assert mm > 100 * ln > 0 and ln > sm, f"expected matmul >> layernorm > softmax, got {mm}, {ln}, {sm}"
