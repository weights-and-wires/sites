from submission import admit_count, blocks_needed, contiguous_waste, paged_cost, paged_waste


def test_blocks_needed():
    got = [blocks_needed(n, 16) for n in (1, 15, 16, 17, 32, 33)]
    assert got == [1, 1, 1, 2, 2, 3], f"expected [1, 1, 1, 2, 2, 3], got {got}"
    assert blocks_needed(7, 4) == 2, f"the 7-token prompt of Figure 6 needs 2 blocks, got {blocks_needed(7, 4)}"


def test_exact_multiple_wastes_nothing():
    assert paged_waste([16, 32, 48], 16) == 0, f"exact multiples must waste 0, got {paged_waste([16, 32, 48], 16)}"
    assert paged_cost(16, 16) == 16, f"expected 16, got {paged_cost(16, 16)}"


def test_paged_waste_values():
    lengths = [1, 16, 17, 32]
    assert paged_waste(lengths, 16) == 30, f"expected 15 + 0 + 15 + 0 = 30, got {paged_waste(lengths, 16)}"
    assert paged_waste([8], 16) == 8, f"Figure 3, request A with 8 tokens: expected 8, got {paged_waste([8], 16)}"
    assert paged_waste([4], 16) == 12, f"Figure 3, request B with 4 tokens: expected 12, got {paged_waste([4], 16)}"


def test_paged_waste_is_below_one_block():
    for B in (1, 4, 16, 64):
        lengths = list(range(1, 200, 7))
        per = paged_waste(lengths, B) / len(lengths)
        assert paged_waste(lengths, B) < B * len(lengths), f"B={B}: waste per request must stay below B, got {per}"


def test_contiguous_waste():
    assert contiguous_waste([8, 4], 2048) == 2040 + 2044, f"expected 4084, got {contiguous_waste([8, 4], 2048)}"
    assert contiguous_waste([], 100) == 0, "no requests waste nothing"
    assert contiguous_waste([64, 64], 64) == 0, "a request that uses its whole reservation wastes nothing"


def test_admit_count_in_order_and_exact_fit():
    assert admit_count([10, 20, 30], 60, lambda n: n) == 3, "10 + 20 + 30 = 60 fits exactly"
    assert admit_count([10, 20, 30], 59, lambda n: n) == 2, "the third request needs 30 and only 29 are left"
    assert admit_count([], 10, lambda n: n) == 0, "no requests"


def test_admit_stops_at_first_misfit():
    got = admit_count([10, 100, 10], 105, lambda n: n)
    assert got == 1, f"the second request doesn't fit, so admission stops; expected 1, got {got}"


def test_contiguous_vs_paged_batch_size():
    lengths = [20] * 10
    contig = admit_count(lengths, 128, lambda n: 64)
    paged = admit_count(lengths, 128, lambda n: paged_cost(n, 16))
    assert contig == 2, f"reserving 64 slots each: expected 2 requests in 128 slots, got {contig}"
    assert paged == 4, f"20 tokens occupy 32 slots in blocks of 16: expected 4 requests, got {paged}"
    assert admit_count(lengths, 128, lambda n: paged_cost(n, 1)) == 6, "block size 1 wastes nothing: expected 6 requests of 20 slots"


def test_larger_blocks_waste_more():
    lengths = [3, 9, 30, 70, 5]
    wastes = [paged_waste(lengths, B) for B in (1, 4, 16, 64)]
    assert wastes[0] == 0, f"block size 1 has no waste, got {wastes[0]}"
    assert wastes == [0, 11, 43, 267], f"unexpected waste values {wastes}"
