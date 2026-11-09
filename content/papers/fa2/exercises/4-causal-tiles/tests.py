import math
from submission import causal_tile_counts, causal_speedup_bound


def brute(N, Br, Bc):
    total = skipped = masked = full = 0
    for i in range(0, N, Br):
        rows = range(i, min(i + Br, N))
        for j in range(0, N, Bc):
            cols = range(j, min(j + Bc, N))
            total += 1
            any_kept = any(c <= r for r in rows for c in cols)
            any_masked = any(c > r for r in rows for c in cols)
            if not any_kept:
                skipped += 1
            elif any_masked:
                masked += 1
            else:
                full += 1
    return total, skipped, masked, full


def test_square_blocks():
    got = causal_tile_counts(1024, 128, 128)
    assert got == (64, 28, 8, 28), f"expected (64, 28, 8, 28), got {got}"


def test_one_masked_tile_per_row_block_when_square():
    total, skipped, masked, full = causal_tile_counts(2048, 64, 64)
    assert masked == 2048 // 64, f"square blocks: expected one masked tile per row block (32), got {masked}"


def test_wide_row_blocks_need_two_masked_tiles():
    got = causal_tile_counts(1024, 128, 64)
    assert got == (128, 56, 16, 56), f"expected (128, 56, 16, 56), got {got}"


def test_matches_brute_force_on_odd_sizes():
    for N, Br, Bc in [(10, 4, 4), (13, 5, 3), (7, 7, 7), (9, 1, 1), (17, 4, 8), (33, 16, 5)]:
        want = brute(N, Br, Bc)
        got = causal_tile_counts(N, Br, Bc)
        assert tuple(got) == want, f"N={N}, Br={Br}, Bc={Bc}: expected {want}, got {got}"


def test_counts_add_up():
    total, skipped, masked, full = causal_tile_counts(1000, 128, 96)
    assert total == math.ceil(1000 / 128) * math.ceil(1000 / 96), f"total tiles wrong: {total}"
    assert skipped + masked + full == total, f"expected the classes to sum to {total}, got {skipped + masked + full}"


def test_speedup_bound():
    got = causal_speedup_bound(1024, 128, 128)
    assert math.isclose(got, 64 / 36), f"expected 64/36 = {64 / 36:.4f}, got {got:.4f}"
    big = causal_speedup_bound(16384, 128, 128)
    assert 1.95 < big < 2.0, f"expected the bound to approach 2 for long sequences, got {big:.4f}"
