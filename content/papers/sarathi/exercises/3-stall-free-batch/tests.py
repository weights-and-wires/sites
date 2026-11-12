from submission import stall_free_batch


def test_decodes_come_first():
    got = stall_free_batch(12, [300, 300], 512)
    assert got == [300, 200], f"12 decodes leave 500 tokens, expected [300, 200], got {got}"


def test_chunk_never_exceeds_remaining():
    got = stall_free_batch(0, [50, 40], 512)
    assert got == [50, 40], f"expected [50, 40], got {got}"


def test_budget_exactly_filled():
    got = stall_free_batch(32, [4096], 512)
    assert got == [480], f"expected [480], got {got}"
    assert 32 + sum(got) == 512, f"the batch must hold exactly 512 tokens, got {32 + sum(got)}"


def test_decodes_exceeding_budget_still_run():
    got = stall_free_batch(600, [100, 100], 512)
    assert got == [0, 0], f"decodes alone exceed the budget, so no prefill: expected [0, 0], got {got}"


def test_priority_order_is_kept():
    got = stall_free_batch(12, [1000, 50], 512)
    assert got == [500, 0], f"the first request must take everything left, expected [500, 0], got {got}"


def test_no_waiting():
    got = stall_free_batch(8, [], 512)
    assert got == [], f"expected [], got {got}"


def test_total_never_exceeds_budget():
    for d in (0, 1, 7, 100, 511, 512):
        got = stall_free_batch(d, [300, 300, 300], 512)
        assert d + sum(got) <= max(512, d), f"d={d}: batch of {d + sum(got)} tokens exceeds the budget; chunks {got}"
        assert len(got) == 3, f"d={d}: need one chunk per waiting request, got {got}"
