from submission import cache_bytes, cache_slots, kv_bytes_per_token


def test_llama2_7b_per_token():
    got = kv_bytes_per_token(32, 32, 128, 2)
    assert got == 524288, f"expected 524288, got {got}"


def test_per_token_counts_keys_and_values_and_layers():
    got = kv_bytes_per_token(2, 3, 4, 1)
    assert got == 2 * 2 * 3 * 4, f"expected 48 (K and V, 2 layers, 3 heads, width 4), got {got}"
    got = kv_bytes_per_token(1, 8, 128, 2)
    assert got == 4096, f"expected 4096, got {got}"


def test_slots_before_and_after_full():
    got = [cache_slots(n, 4, 8) for n in (0, 5, 12, 13, 1000)]
    assert got == [0, 5, 12, 12, 12], f"expected [0, 5, 12, 12, 12], got {got}"


def test_full_cache_grows_linearly():
    got = cache_bytes(4_000_000, 524288)
    assert got == 4_000_000 * 524288, f"expected {4_000_000 * 524288}, got {got}"
    assert got / 2**30 == 4_000_000 * 524288 / 2**30, f"expected about 1953.1 GiB, got {got / 2**30}"


def test_streaming_cache_is_constant():
    a = cache_bytes(4096, 524288, sinks=4, window=2044)
    b = cache_bytes(4_000_000, 524288, sinks=4, window=2044)
    assert a == b == 2048 * 524288, f"expected {2048 * 524288} for both, got {a} and {b}"
    assert b / 2**30 == 1.0, f"expected 1.0 GiB, got {b / 2**30}"


def test_short_text_streaming_equals_full():
    a = cache_bytes(100, 1000, sinks=4, window=2044)
    b = cache_bytes(100, 1000)
    assert a == b == 100_000, f"before the cache fills both hold 100 tokens: expected 100000, got {a} and {b}"


def test_zero_sinks_is_a_window():
    got = cache_bytes(10_000, 10, sinks=0, window=100)
    assert got == 1000, f"expected 1000, got {got}"
