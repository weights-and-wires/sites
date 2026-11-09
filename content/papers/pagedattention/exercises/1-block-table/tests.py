import math

from submission import BlockAllocator, BlockTable

FIG6_ORDER = [7, 1, 3, 0, 2, 4, 5, 6, 8]


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    except Exception as e:  # a different exception type
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    return False


def test_allocator_order_and_counts():
    a = BlockAllocator(3)
    got = [a.allocate(), a.allocate()]
    assert got == [0, 1], f"expected the front of the free list [0, 1], got {got}"
    assert a.num_free == 1, f"expected 1 free block, got {a.num_free}"
    assert a.ref_count(0) == 1 and a.ref_count(2) == 0, f"expected counts 1 and 0, got {a.ref_count(0)} and {a.ref_count(2)}"


def test_free_goes_to_the_back():
    a = BlockAllocator(3)
    x, y = a.allocate(), a.allocate()
    a.free(x)
    got = [a.allocate(), a.allocate()]
    assert got == [2, 0], f"a freed block rejoins the back of the free list; expected [2, 0], got {got}"


def test_out_of_memory():
    a = BlockAllocator(2)
    a.allocate(), a.allocate()
    assert raises(MemoryError, a.allocate), "expected MemoryError when no block is free"


def test_reference_counting():
    a = BlockAllocator(2)
    b = a.allocate()
    a.incref(b)
    assert a.ref_count(b) == 2, f"expected count 2 after incref, got {a.ref_count(b)}"
    a.free(b)
    assert a.ref_count(b) == 1 and a.num_free == 1, f"one free of a shared block must keep it allocated; count {a.ref_count(b)}, free {a.num_free}"
    a.free(b)
    assert a.ref_count(b) == 0 and a.num_free == 2, f"the last free must release it; count {a.ref_count(b)}, free {a.num_free}"


def test_double_free_and_bad_incref():
    a = BlockAllocator(2)
    b = a.allocate()
    a.free(b)
    assert raises(ValueError, a.free, b), "expected ValueError when freeing a free block"
    assert raises(ValueError, a.incref, b), "expected ValueError when incref-ing a free block"
    assert a.num_free == 2, f"a failed free must not change the free list; expected 2, got {a.num_free}"


def test_figure_6_trace():
    a = BlockAllocator(9, order=FIG6_ORDER)
    t = BlockTable(a, 4)
    placed = [t.append_token() for _ in range(7)]
    assert t.blocks == [7, 1], f"a 7-token prompt with block size 4 maps to physical blocks [7, 1]; got {t.blocks}"
    assert placed[0] == (7, 0) and placed[3] == (7, 3) and placed[4] == (1, 0) and placed[6] == (1, 2), f"unexpected placements {placed}"
    assert t.wasted_slots == 1, f"expected 1 reserved slot, got {t.wasted_slots}"
    assert t.append_token() == (1, 3), "the first decoding step fills the last slot of block 1"
    assert t.blocks == [7, 1], f"no new block yet; got {t.blocks}"
    assert t.append_token() == (3, 0), "the second decoding step needs a new block, physical block 3"
    assert t.blocks == [7, 1, 3], f"expected [7, 1, 3], got {t.blocks}"


def test_no_eager_allocation():
    a = BlockAllocator(10)
    t = BlockTable(a, 4)
    assert t.blocks == [] and a.num_free == 10, "an empty table holds no block"
    for _ in range(8):
        t.append_token()
    assert len(t.blocks) == 2 and t.wasted_slots == 0, f"8 tokens fill exactly 2 blocks; got {len(t.blocks)} blocks, {t.wasted_slots} wasted"
    t.append_token()
    assert len(t.blocks) == 3 and t.wasted_slots == 3, f"the 9th token opens block 3; got {len(t.blocks)} blocks, {t.wasted_slots} wasted"


def test_waste_stays_inside_the_last_block():
    for B in (1, 4, 16):
        a = BlockAllocator(100)
        t = BlockTable(a, B)
        for n in range(1, 41):
            t.append_token()
            assert len(t.blocks) == math.ceil(n / B), f"B={B}, n={n}: expected {math.ceil(n / B)} blocks, got {len(t.blocks)}"
            assert 0 <= t.wasted_slots < B, f"B={B}, n={n}: waste must be in [0, {B}), got {t.wasted_slots}"


def test_translate():
    a = BlockAllocator(9, order=FIG6_ORDER)
    t = BlockTable(a, 4)
    for _ in range(9):
        t.append_token()
    assert t.translate(0) == (7, 0), f"position 0: expected (7, 0), got {t.translate(0)}"
    assert t.translate(5) == (1, 1), f"position 5: expected (1, 1), got {t.translate(5)}"
    assert t.translate(8) == (3, 0), f"position 8: expected (3, 0), got {t.translate(8)}"
    assert raises(IndexError, t.translate, 9), "position 9 is past the last token"
    assert raises(IndexError, t.translate, -1), "a negative position is out of range"


def test_two_sequences_interleave_without_contiguity():
    a = BlockAllocator(8)
    t1, t2 = BlockTable(a, 2), BlockTable(a, 2)
    for _ in range(4):
        t1.append_token()
        t2.append_token()
    assert t1.blocks == [0, 2] and t2.blocks == [1, 3], f"expected [0, 2] and [1, 3], got {t1.blocks} and {t2.blocks}"
    assert not set(t1.blocks) & set(t2.blocks), "two sequences must never share a block here"


def test_release_returns_blocks():
    a = BlockAllocator(6)
    t = BlockTable(a, 4)
    for _ in range(10):
        t.append_token()
    assert a.num_free == 3, f"10 tokens use 3 of 6 blocks; expected 3 free, got {a.num_free}"
    t.release()
    assert a.num_free == 6 and t.blocks == [] and t.num_tokens == 0, f"release must free all; free {a.num_free}, blocks {t.blocks}, tokens {t.num_tokens}"
    assert all(a.ref_count(b) == 0 for b in range(6)), "every block must read 0 references after release"
