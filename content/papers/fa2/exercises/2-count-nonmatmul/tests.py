import math
from submission import nonmatmul_ops, matmul_flops, nonmatmul_time_share


def test_matmul_flops():
    assert matmul_flops(1024, 64) == 4 * 1024 * 1024 * 64, f"expected {4 * 1024 * 1024 * 64}, got {matmul_flops(1024, 64)}"
    assert matmul_flops(10, 3) == 1200, f"expected 1200, got {matmul_flops(10, 3)}"


def test_fa1_single_tile():
    # One tile: 4*Br*Bc + 9*Br + 3*Br*d = 4*4*4 + 9*4 + 3*4*2 = 124.
    got = nonmatmul_ops(4, 2, 4, 4, "fa1")
    assert got == 124, f"expected 124, got {got}"


def test_fa2_single_tile():
    # One tile: 4*16 + 5*4 + 4*2 = 92, plus the final scale 4*2 + 2*4 = 16.
    got = nonmatmul_ops(4, 2, 4, 4, "fa2")
    assert got == 108, f"expected 108, got {got}"


def test_fa2_divides_once_per_row_block_not_per_tile():
    # Doubling the number of K/V blocks adds only per-tile work in FA2.
    one = nonmatmul_ops(8, 4, 8, 8, "fa2")
    two = nonmatmul_ops(16, 4, 8, 8, "fa2")  # Tr=2, Tc=2
    per_tile = 4 * 8 * 8 + 5 * 8 + 8 * 4
    final = 8 * 4 + 2 * 8
    assert one == per_tile + final, f"expected {per_tile + final}, got {one}"
    assert two == 2 * (2 * per_tile + final), f"expected {2 * (2 * per_tile + final)}, got {two}"


def test_fa1_scales_per_tile():
    one = nonmatmul_ops(8, 4, 8, 8, "fa1")
    four = nonmatmul_ops(16, 4, 8, 8, "fa1")
    assert four == 4 * one, f"FA1 has no per-row-block term, so 4 tiles cost 4x one tile: expected {4 * one}, got {four}"


def test_partial_blocks_round_up():
    got = nonmatmul_ops(10, 2, 4, 4, "fa1")  # Tr = Tc = 3
    want = 9 * (4 * 16 + 9 * 4 + 3 * 4 * 2)
    assert got == want, f"expected {want} (3 x 3 tiles of full size), got {got}"


def test_fa2_does_fewer_ops_than_fa1():
    for N, d, Br, Bc in [(1024, 64, 128, 64), (4096, 128, 128, 128), (2048, 64, 64, 64)]:
        a, b = nonmatmul_ops(N, d, Br, Bc, "fa1"), nonmatmul_ops(N, d, Br, Bc, "fa2")
        assert b < a, f"N={N}, d={d}: expected FA2 < FA1, got fa1={a}, fa2={b}"


def test_time_share():
    got1 = nonmatmul_time_share(1024, 64, 128, 64, "fa1")
    got2 = nonmatmul_time_share(1024, 64, 128, 64, "fa2")
    nm1 = 16 * nonmatmul_ops(1024, 64, 128, 64, "fa1")
    want1 = nm1 / (nm1 + 4 * 1024 * 1024 * 64)
    assert math.isclose(got1, want1), f"expected {want1:.6f}, got {got1:.6f}"
    assert got2 < got1, f"FA2 share {got2:.4f} should be below FA1 share {got1:.4f}"
    assert nonmatmul_time_share(1024, 64, 128, 64, "fa2", ratio=1) < got2, "a smaller ratio must give a smaller share"


def test_unknown_version_raises():
    try:
        nonmatmul_ops(8, 4, 4, 4, "fa3")
    except ValueError:
        return
    assert False, "expected ValueError for an unknown version"
