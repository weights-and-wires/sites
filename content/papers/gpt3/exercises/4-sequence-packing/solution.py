import numpy as np


def pack_documents(docs, n_ctx, eot):
    """Pack token lists into an (n_sequences, n_ctx) int array, dropping the short remainder."""
    stream = []
    for doc in docs:
        stream.extend(doc)
        stream.append(eot)
    n_seq = len(stream) // n_ctx
    return np.array(stream[: n_seq * n_ctx], dtype=int).reshape(n_seq, n_ctx)


def document_ids(seq, eot):
    """Index of the document each position belongs to; an eot token ends its own document."""
    seq = np.asarray(seq)
    ends = (seq == eot).astype(int)
    return np.concatenate([[0], np.cumsum(ends)[:-1]]).astype(int)


def cross_document_pairs(seq, eot):
    """Count causal (query i, key j <= i) pairs whose positions are in different documents."""
    ids = document_ids(seq, eot)
    total = 0
    for i in range(len(ids)):
        total += int(np.sum(ids[: i + 1] != ids[i]))
    return total
