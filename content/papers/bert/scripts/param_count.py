"""Parameter count of BERT-BASE and BERT-LARGE (Section 3, footnote 3 for the
feed-forward size 4H), computed from the architecture.

The paper gives a 30,000-token WordPiece vocabulary (Section 3). The released
English uncased checkpoints use 30,522 tokens, so the script prints both.

Run: python3 scripts/param_count.py
"""


def count(L, H, vocab, max_pos=512, n_seg=2, heads_and_pooler=True):
    ffn = 4 * H
    attn = 4 * (H * H + H)                      # Q, K, V, output projections with biases
    ffn_p = H * ffn + ffn + ffn * H + H         # two linear layers with biases
    ln = 2 * (2 * H)                            # two layer norms, scale and shift
    block = attn + ffn_p + ln
    emb_tok, emb_pos, emb_seg, emb_ln = vocab * H, max_pos * H, n_seg * H, 2 * H
    embeddings = emb_tok + emb_pos + emb_seg + emb_ln
    pooler = H * H + H
    mlm_head = H * H + H + 2 * H + vocab        # transform, layer norm, output bias (weights tied)
    nsp_head = H * 2 + 2
    encoder = L * block + embeddings
    total = encoder + (pooler + mlm_head + nsp_head if heads_and_pooler else 0)
    return dict(block=block, embeddings=embeddings, tok=emb_tok, encoder=encoder, pooler=pooler,
                mlm_head=mlm_head, nsp_head=nsp_head, total=total)


for name, L, H, A in (("BERT-BASE", 12, 768, 12), ("BERT-LARGE", 24, 1024, 16)):
    print(f"{name}: L={L}, H={H}, A={A}, head size {H // A}")
    for vocab in (30000, 30522):
        c = count(L, H, vocab)
        print(f"  vocab {vocab}: block {c['block']:,}; embeddings {c['embeddings']:,}; "
              f"encoder (blocks + embeddings) {c['encoder']:,}; "
              f"with pooler, MLM and NSP heads {c['total']:,}")
    c = count(L, H, 30522)
    print(f"  all {L} blocks: {L * c['block']:,} ({L * c['block'] / c['total']:.1%} of the total at vocab 30522)")
    print(f"  token embeddings: {c['tok']:,} ({c['tok'] / c['total']:.1%})")
