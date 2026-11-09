"""Packing documents into fixed-length training sequences (Appendix B).

Appendix B says GPT-3 trains on sequences of the full n_ctx = 2048 tokens,
packing several documents into one sequence, delimited by an end-of-text token,
with no special attention mask. The paper doesn't say what happens to a
document that crosses a sequence boundary or to a short final remainder; this
script cuts one token stream into equal pieces and drops the remainder.
Token ids and document lengths here are invented.

Run: python3 scripts/packing_example.py
"""

import numpy as np

EOT = 0


def pack(docs, n_ctx):
    stream = []
    for d in docs:
        stream += list(d) + [EOT]
    n_seq = len(stream) // n_ctx
    return np.array(stream[: n_seq * n_ctx]).reshape(n_seq, n_ctx), len(stream) - n_seq * n_ctx


def doc_ids(seq):
    """The EOT token belongs to the document it ends, so count EOTs strictly before i."""
    return np.concatenate([[0], np.cumsum(np.asarray(seq) == EOT)[:-1]])


def cross_pairs(seq):
    ids = doc_ids(seq)
    n = len(seq)
    return sum(int(ids[j] != ids[i]) for i in range(n) for j in range(i + 1))


docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
seqs, dropped = pack(docs, 4)
print("docs:", docs)
print("packed with n_ctx = 4:\n", seqs, "| dropped tokens:", dropped)
for s in seqs:
    n = len(s)
    print(f"  {s.tolist()} doc ids {doc_ids(s).tolist()} cross-document pairs {cross_pairs(s)} of {n * (n + 1) // 2}")

rng = np.random.default_rng(0)
lengths = np.maximum(rng.exponential(600, size=60).astype(int), 1)
docs = [rng.integers(1, 50000, size=n).tolist() for n in lengths]
seqs, dropped = pack(docs, 2048)
total = 2048 * 2049 // 2
cross = [cross_pairs(s) for s in seqs]
print()
print(f"{len(docs)} documents, mean length {lengths.mean():.0f} -> {len(seqs)} sequences, {dropped} tokens dropped")
print("EOT tokens per sequence:", [int((s == EOT).sum()) for s in seqs[:6]], "...")
print(f"share of causal (query, key) pairs that cross a document boundary: {100 * np.mean(cross) / total:.1f}%")
