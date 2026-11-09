import math
from submission import min_tile_for_ridge, tiled_matmul_intensity


def test_tile_one():
    got = tiled_matmul_intensity(4096, 1, 2)
    want = 2 * 4096 ** 3 / (2 * (2 * 4096 ** 3 + 4096 ** 2))
    assert math.isclose(got, want), f"tile 1: expected {want:.6f}, got {got}"
    assert abs(got - 0.5) < 1e-3, f"tile 1 in BF16 should be close to 0.5 FLOPs/byte, got {got}"


def test_whole_matrix_is_compulsory_traffic():
    got = tiled_matmul_intensity(512, 512, 4)
    assert math.isclose(got, 512 / 6), f"tile n should give the compulsory intensity n/6 = {512 / 6:.4f}, got {got}"


def test_known_values():
    got = tiled_matmul_intensity(4096, 128, 2)
    assert abs(got - 63.02) < 0.01, f"n=4096, tile=128, BF16: expected 63.02, got {got}"
    got = tiled_matmul_intensity(4096, 16, 2)
    assert abs(got - 7.98) < 0.01, f"n=4096, tile=16, BF16: expected 7.98, got {got}"


def test_monotone_in_tile():
    vals = [tiled_matmul_intensity(1024, t, 4) for t in (1, 2, 4, 8, 16, 64, 256, 1024)]
    assert all(a < b for a, b in zip(vals, vals[1:])), f"intensity should rise with tile size, got {vals}"


def test_min_tile_is_minimal():
    n, b, ridge = 4096, 2, 100.0
    t = min_tile_for_ridge(n, ridge, b)
    assert t is not None, "100 FLOPs/byte is reachable for n=4096 in BF16"
    assert tiled_matmul_intensity(n, t, b) >= ridge, f"tile {t} gives {tiled_matmul_intensity(n, t, b)}, below the ridge {ridge}"
    assert tiled_matmul_intensity(n, t - 1, b) < ridge, f"tile {t - 1} already reaches the ridge, so {t} is not the smallest"


def test_min_tile_known_value():
    t = min_tile_for_ridge(8192, 295.2, 2)
    assert t == 613, f"H100 BF16 ridge 295.2 at n=8192: expected tile 613, got {t}"


def test_unreachable_returns_none():
    got = min_tile_for_ridge(256, 1000.0, 2)
    assert got is None, f"n=256 tops out near {tiled_matmul_intensity(256, 256, 2):.1f}, so 1000 is unreachable; expected None, got {got}"
