import numpy as np
from submission import embed_inputs, pack_pair

CLS, SEP = 1, 2


def test_pack_pair_layout():
    ids, segs = pack_pair([10, 11, 12], [20, 21], CLS, SEP)
    assert list(ids) == [1, 10, 11, 12, 2, 20, 21, 2], f"expected [1, 10, 11, 12, 2, 20, 21, 2], got {list(ids)}"
    assert list(segs) == [0, 0, 0, 0, 0, 1, 1, 1], f"expected [0, 0, 0, 0, 0, 1, 1, 1], got {list(segs)}"


def test_lengths_for_other_shapes():
    ids, segs = pack_pair([5], [6, 7, 8, 9], CLS, SEP)
    assert len(ids) == 1 + 1 + 1 + 4 + 1, f"expected 8 ids, got {len(ids)}: {list(ids)}"
    assert list(segs) == [0, 0, 0, 1, 1, 1, 1, 1], f"expected [0, 0, 0, 1, 1, 1, 1, 1], got {list(segs)}"
    assert ids[0] == CLS and ids[-1] == SEP and ids[2] == SEP, f"[CLS] must lead and [SEP] must close each sentence, got {list(ids)}"


def test_inputs_are_not_modified():
    a, b = [10, 11], [20]
    pack_pair(a, b, CLS, SEP)
    assert a == [10, 11] and b == [20], f"inputs were modified: {a}, {b}"


def tables():
    tok = np.zeros((30, 2))
    tok[CLS], tok[SEP], tok[10], tok[11], tok[12], tok[20], tok[21] = (
        [1, 0], [0, 1], [2, 0], [0, 2], [2, 2], [3, 0], [0, 3])
    seg = np.array([[0.5, 0.5], [-0.5, -0.5]])
    pos = np.arange(32, dtype=float).reshape(16, 2) / 10
    return tok, seg, pos


def test_embedding_sum_from_the_page():
    tok, seg, pos = tables()
    ids, segs = pack_pair([10, 11, 12], [20, 21], CLS, SEP)
    x = embed_inputs(ids, segs, tok, seg, pos)
    assert x.shape == (8, 2), f"expected shape (8, 2), got {x.shape}"
    assert np.allclose(x[4], [1.3, 2.4]), f"expected position 4 ([SEP]) = [1.3, 2.4], got {x[4]}"
    assert np.allclose(x[5], [3.5, 0.6]), f"expected position 5 = [3.5, 0.6], got {x[5]}"


def test_same_token_differs_by_position_and_segment():
    tok, seg, pos = tables()
    ids, segs = pack_pair([10], [10], CLS, SEP)
    x = embed_inputs(ids, segs, tok, seg, pos)
    assert not np.allclose(x[1], x[3]), f"the same token at positions 1 and 3 must differ, got {x[1]} and {x[3]}"
    want = x[3] - x[1]
    expect = (pos[3] - pos[1]) + (seg[1] - seg[0])
    assert np.allclose(want, expect), f"expected difference {expect}, got {want}"


def test_position_uses_the_position_index_not_the_token_id():
    tok = np.zeros((30, 2))
    seg = np.zeros((2, 2))
    pos = np.arange(40, dtype=float).reshape(20, 2)
    x = embed_inputs([7, 7, 7], [0, 0, 0], tok, seg, pos)
    assert np.allclose(x, pos[:3]), f"expected rows 0, 1, 2 of the position table, got {x}"
