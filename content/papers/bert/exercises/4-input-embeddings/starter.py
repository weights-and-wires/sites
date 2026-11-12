import numpy as np


def pack_pair(a, b, cls_id, sep_id):
    """Return (ids, segment_ids) for [CLS] a [SEP] b [SEP]."""
    raise NotImplementedError


def embed_inputs(ids, segment_ids, tok_table, seg_table, pos_table):
    """Return token + segment + position embeddings, shape (T, d)."""
    raise NotImplementedError
