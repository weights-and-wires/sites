import numpy as np


def pack_pair(a, b, cls_id, sep_id):
    """Return (ids, segment_ids) for [CLS] a [SEP] b [SEP]."""
    ids = [cls_id] + list(a) + [sep_id] + list(b) + [sep_id]
    segment_ids = [0] * (len(a) + 2) + [1] * (len(b) + 1)
    return ids, segment_ids


def embed_inputs(ids, segment_ids, tok_table, seg_table, pos_table):
    """Return token + segment + position embeddings, shape (T, d)."""
    ids = np.asarray(ids)
    segment_ids = np.asarray(segment_ids)
    T = len(ids)
    return tok_table[ids] + seg_table[segment_ids] + pos_table[np.arange(T)]
