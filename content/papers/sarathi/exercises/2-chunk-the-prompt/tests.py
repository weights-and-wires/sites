from submission import chunk_sizes, kv_tokens_reread


def test_even_split():
    got = chunk_sizes(4096, 512)
    assert got == [512] * 8, f"expected eight chunks of 512, got {got}"


def test_remainder_goes_last():
    got = chunk_sizes(1000, 300)
    assert got == [300, 300, 300, 100], f"expected [300, 300, 300, 100], got {got}"


def test_short_prompt_is_one_chunk():
    got = chunk_sizes(100, 512)
    assert got == [100], f"expected [100], got {got}"


def test_exact_budget_has_no_empty_chunk():
    got = chunk_sizes(512, 512)
    assert got == [512], f"expected [512], got {got}"
    got = chunk_sizes(1024, 512)
    assert got == [512, 512], f"expected [512, 512], got {got}"


def test_sizes_sum_and_are_positive():
    for p, b in ((1, 1), (777, 64), (4097, 512), (9, 4)):
        got = chunk_sizes(p, b)
        assert sum(got) == p, f"({p}, {b}): chunks {got} must sum to {p}"
        assert all(0 < c <= b for c in got), f"({p}, {b}): every chunk must be in 1..{b}, got {got}"


def test_reread_equal_chunks():
    got = kv_tokens_reread([512] * 8)
    assert got == 14336, f"512 * 28 = 14336, got {got}"


def test_reread_uneven_chunks():
    got = kv_tokens_reread([300, 300, 300, 100])
    assert got == 1800, f"0 + 300 + 600 + 900 = 1800, got {got}"


def test_single_chunk_rereads_nothing():
    got = kv_tokens_reread([100])
    assert got == 0, f"one chunk has no earlier chunks, expected 0, got {got}"
    got = kv_tokens_reread([])
    assert got == 0, f"no chunks, expected 0, got {got}"


def test_smaller_chunks_reread_more():
    a = kv_tokens_reread(chunk_sizes(4096, 1024))
    b = kv_tokens_reread(chunk_sizes(4096, 256))
    assert b > a, f"256-token chunks must re-read more than 1024-token chunks: {b} vs {a}"
    assert (a, b) == (6144, 30720), f"expected (6144, 30720), got {(a, b)}"
