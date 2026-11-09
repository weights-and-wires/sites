from submission import kv_cache_bytes_per_token, max_concurrent_sequences

GIB = 2**30
GPU = 80 * GIB
WEIGHTS = 13_476_831_232  # 6,738,415,616 parameters in fp16
L, D = 32, 128


def test_mha_7b_per_token():
    got = kv_cache_bytes_per_token(L, 32, D)
    assert got == 524288, f"MHA at 7B scale should be 524288 bytes (512 KiB) per token, got {got}"


def test_gqa8_and_mqa_per_token():
    gqa = kv_cache_bytes_per_token(L, 8, D)
    mqa = kv_cache_bytes_per_token(L, 1, D)
    assert gqa == 131072, f"GQA-8 should be 131072 bytes per token, got {gqa}"
    assert mqa == 16384, f"MQA should be 16384 bytes per token, got {mqa}"


def test_counts_keys_and_values():
    got = kv_cache_bytes_per_token(1, 1, 1, bytes_per_value=1)
    assert got == 2, f"one layer, one head, one element, one byte stores a key and a value: expected 2, got {got}"


def test_depends_on_layers_and_precision():
    base = kv_cache_bytes_per_token(4, 2, 8, 2)
    assert base == 2 * 4 * 2 * 8 * 2, f"expected {2 * 4 * 2 * 8 * 2}, got {base}"
    fp32 = kv_cache_bytes_per_token(4, 2, 8, 4)
    assert fp32 == 2 * base, f"fp32 should use twice the bytes of fp16: expected {2 * base}, got {fp32}"
    deeper = kv_cache_bytes_per_token(8, 2, 8, 2)
    assert deeper == 2 * base, f"doubling the layers should double the cache: expected {2 * base}, got {deeper}"


def test_reduction_is_h_over_g():
    mha = kv_cache_bytes_per_token(L, 32, D)
    for g in (16, 8, 4, 1):
        got = mha // kv_cache_bytes_per_token(L, g, D)
        assert got == 32 // g, f"GQA-{g} should cut the cache by {32 // g}x, got {got}x"


def test_sequences_that_fit():
    mha = max_concurrent_sequences(GPU, WEIGHTS, 4096, L, 32, D)
    gqa = max_concurrent_sequences(GPU, WEIGHTS, 4096, L, 8, D)
    mqa = max_concurrent_sequences(GPU, WEIGHTS, 4096, L, 1, D)
    assert (mha, gqa, mqa) == (33, 134, 1079), f"expected (33, 134, 1079) sequences for MHA, GQA-8, MQA; got {(mha, gqa, mqa)}"


def test_rounds_down_and_returns_int():
    got = max_concurrent_sequences(10, 0, 1, 1, 1, 3, 1)  # 6 bytes per sequence
    assert got == 1 and isinstance(got, int), f"10 bytes free, 6 per sequence: expected the int 1, got {got!r}"
    got = max_concurrent_sequences(11, 5, 1, 1, 1, 3, 1)  # 6 free, 6 per sequence
    assert got == 1, f"exactly one sequence fits in 6 bytes: expected 1, got {got}"
    got = max_concurrent_sequences(10, 5, 1, 1, 1, 3, 1)  # 5 free
    assert got == 0, f"5 bytes free is not enough for a 6-byte sequence: expected 0, got {got}"
