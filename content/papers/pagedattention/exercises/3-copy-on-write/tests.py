import math

from submission import BlockAllocator, SeqTable

FIG8_ORDER = [7, 1, 3, 0, 2, 4, 5, 6, 8]


def prefill(alloc, B, n):
    t = SeqTable(alloc, B)
    for _ in range(n):
        t.append_token()
    return t


def test_figure_8_trace():
    alloc = BlockAllocator(9, order=FIG8_ORDER)
    a1 = prefill(alloc, 4, 7)
    assert a1.blocks == [7, 1], f"the prompt maps to physical blocks [7, 1], got {a1.blocks}"
    a2 = a1.fork()
    assert a2.blocks == [7, 1] and a2.num_tokens == 7, f"the fork shares the table; got {a2.blocks}, {a2.num_tokens} tokens"
    assert alloc.ref_count(7) == 2 and alloc.ref_count(1) == 2, f"both blocks have count 2; got {alloc.ref_count(7)}, {alloc.ref_count(1)}"
    assert alloc.copies == [], f"a fork copies nothing; got {alloc.copies}"
    placed1 = a1.append_token()
    assert alloc.copies == [(1, 3)], f"A1 copies block 1 into block 3; got {alloc.copies}"
    assert placed1 == (3, 3), f"A1 writes offset 3 of block 3; got {placed1}"
    assert alloc.ref_count(1) == 1, f"the copy lowers block 1 to count 1; got {alloc.ref_count(1)}"
    placed2 = a2.append_token()
    assert alloc.copies == [(1, 3)], f"A2 now owns block 1 alone, so no second copy; got {alloc.copies}"
    assert placed2 == (1, 3), f"A2 writes in place at offset 3 of block 1; got {placed2}"
    assert a1.blocks == [7, 3] and a2.blocks == [7, 1], f"expected [7, 3] and [7, 1], got {a1.blocks} and {a2.blocks}"
    assert alloc.ref_count(7) == 2, f"the full prompt block stays shared; got {alloc.ref_count(7)}"


def test_samples_never_write_the_same_slot():
    alloc = BlockAllocator(40)
    base = prefill(alloc, 4, 6)
    seqs = [base, base.fork(), base.fork()]
    slots = [s.append_token() for s in seqs]
    assert len(set(slots)) == 3, f"three samples wrote to slots {slots}; each must own its slot"


def test_full_last_block_needs_no_copy():
    alloc = BlockAllocator(10)
    a = prefill(alloc, 4, 8)
    b = a.fork()
    pa, pb = a.append_token(), b.append_token()
    assert alloc.copies == [], f"a full block is never written, so nothing is copied; got {alloc.copies}"
    assert pa[1] == 0 and pb[1] == 0 and pa[0] != pb[0], f"each sample opens its own new block; got {pa} and {pb}"
    assert a.blocks[:2] == b.blocks[:2] and alloc.ref_count(a.blocks[0]) == 2, "the two full blocks stay shared"


def test_exclusive_block_is_written_in_place():
    alloc = BlockAllocator(10)
    a = prefill(alloc, 4, 6)
    last = a.blocks[-1]
    a.append_token()
    assert a.blocks[-1] == last and alloc.copies == [], f"count 1 means write in place; got {a.blocks}, copies {alloc.copies}"


def test_parallel_sampling_block_count():
    # scripts/sharing_example.py: P=100, T=50, n=4, B=16 -> 22 blocks with sharing, 40 without.
    alloc = BlockAllocator(100)
    base = prefill(alloc, 16, 100)
    seqs = [base] + [base.fork() for _ in range(3)]
    for s in seqs:
        for _ in range(50):
            s.append_token()
    used = 100 - alloc.num_free
    assert used == 22, f"expected 22 physical blocks with sharing, got {used}"
    assert sum(len(s.blocks) for s in seqs) == 40, "without sharing the four tables would hold 40 blocks"
    assert len(alloc.copies) == 3, f"3 of the 4 samples copy the partial prompt block; got {len(alloc.copies)} copies"


def test_release_frees_only_unreferenced_blocks():
    alloc = BlockAllocator(10)
    a = prefill(alloc, 4, 8)
    b = a.fork()
    a.release()
    assert alloc.num_free == 8 and alloc.ref_count(b.blocks[0]) == 1, f"B still holds both blocks; free {alloc.num_free}"
    b.release()
    assert alloc.num_free == 10, f"expected all 10 blocks free, got {alloc.num_free}"


def test_beam_candidate_dropped():
    alloc = BlockAllocator(20)
    a = prefill(alloc, 4, 10)
    b = a.fork()
    c = b.fork()
    c.append_token()
    b.release()
    assert alloc.ref_count(a.blocks[0]) == 2, f"blocks shared by A and C keep count 2; got {alloc.ref_count(a.blocks[0])}"
    a.release()
    c.release()
    assert alloc.num_free == 20, f"expected every block freed, got {alloc.num_free} free"
