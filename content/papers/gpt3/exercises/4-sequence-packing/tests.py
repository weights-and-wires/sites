import numpy as np
from submission import cross_document_pairs, document_ids, pack_documents

DOCS = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]


def test_pack_values():
    got = pack_documents(DOCS, 4, 0)
    want = np.array([[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]])
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"


def test_last_document_gets_an_eot():
    got = pack_documents([[1, 2, 3]], 4, 0)
    assert np.array_equal(got, np.array([[1, 2, 3, 0]])), f"expected [[1 2 3 0]], got {got}"


def test_short_remainder_is_dropped():
    got = pack_documents([[1, 2, 3, 4, 5, 6, 7, 8, 9]], 4, 0)
    assert got.shape == (2, 4), f"10 stream tokens make 2 sequences of 4 and drop 2; got shape {got.shape}"
    assert np.array_equal(got[1], [5, 6, 7, 8]), f"expected [5 6 7 8] in row 1, got {got[1]}"


def test_no_full_sequence():
    got = pack_documents([[1]], 4, 0)
    assert got.shape == (0, 4), f"expected shape (0, 4), got {got.shape}"


def test_document_may_span_sequences():
    got = pack_documents([[1, 2, 3, 4, 5], [6]], 4, 0)
    want = np.array([[1, 2, 3, 4], [5, 0, 6, 0]])
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"


def test_document_ids():
    got = document_ids([4, 5, 0, 6], 0)
    assert np.array_equal(got, [0, 0, 0, 1]), f"expected [0 0 0 1], got {got}"
    got = document_ids([7, 0, 8, 0, 9], 0)
    assert np.array_equal(got, [0, 0, 1, 1, 2]), f"expected [0 0 1 1 2], got {got}"


def test_cross_pairs_small():
    got = cross_document_pairs([4, 5, 0, 6], 0)
    assert got == 3, f"position 3 attends to 3 keys in the earlier document; expected 3, got {got}"
    got = cross_document_pairs([1, 2, 3, 0], 0)
    assert got == 0, f"one document has no crossing pairs; expected 0, got {got}"


def test_cross_pairs_counts_each_pair_once():
    got = cross_document_pairs([7, 0, 8, 0, 9], 0)
    # doc 0 = positions 0-1, doc 1 = positions 2-3, doc 2 = position 4.
    # query 2 and 3 see 2 earlier-document keys each; query 4 sees 4.
    assert got == 8, f"expected 8, got {got}"


def test_a_document_mask_would_remove_them_all():
    seq = np.array([4, 5, 0, 6, 7, 0, 8, 9])
    ids = document_ids(seq, 0)
    n = len(seq)
    same = sum(int(ids[j] == ids[i]) for i in range(n) for j in range(i + 1))
    got = cross_document_pairs(seq, 0)
    assert same + got == n * (n + 1) // 2, f"same-document plus cross-document pairs must cover all {n * (n + 1) // 2} causal pairs; got {same} + {got}"
