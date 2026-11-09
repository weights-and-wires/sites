import numpy as np


def pack_documents(docs, n_ctx, eot):
    """Pack token lists into an (n_sequences, n_ctx) int array, dropping the short remainder."""
    raise NotImplementedError


def document_ids(seq, eot):
    """Index of the document each position belongs to; an eot token ends its own document."""
    raise NotImplementedError


def cross_document_pairs(seq, eot):
    """Count causal (query i, key j <= i) pairs whose positions are in different documents."""
    raise NotImplementedError
