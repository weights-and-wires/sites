#!/usr/bin/env python3
"""Count Whisper parameters from the Table 1 dimensions and compare with the table.

The paper gives layers, width, and heads (Table 1) but not the vocabulary size.
The count uses the released multilingual tokenizer, 51,865 tokens, and the
released code's layout: key projections have no bias, the decoder context is
448 tokens, and the output layer is tied to the token embedding (Section 2.2).
"""

N_VOCAB = 51865
N_MELS = 80
N_TEXT_CTX = 448
TABLE_1 = {  # name: (layers, width, heads, parameters in millions)
    "Tiny": (4, 384, 6, 39),
    "Base": (6, 512, 8, 74),
    "Small": (12, 768, 12, 244),
    "Medium": (24, 1024, 16, 769),
    "Large": (32, 1280, 20, 1550),
}


def attention(d):
    return 4 * d * d + 3 * d  # q, k, v, out matrices; k has no bias


def mlp(d):
    return d * 4 * d + 4 * d + 4 * d * d + d


def layer_norm(d):
    return 2 * d


def count(layers, d):
    enc_stem = N_MELS * d * 3 + d + d * d * 3 + d
    enc_block = attention(d) + mlp(d) + 2 * layer_norm(d)
    encoder = enc_stem + layers * enc_block + layer_norm(d)
    dec_block = 2 * attention(d) + mlp(d) + 3 * layer_norm(d)
    decoder = N_VOCAB * d + N_TEXT_CTX * d + layers * dec_block + layer_norm(d)
    return encoder, decoder


if __name__ == "__main__":
    print(f"{'model':7s} {'encoder':>9s} {'decoder':>9s} {'total':>9s} {'Table 1':>8s} {'diff':>7s}")
    for name, (layers, d, heads, table) in TABLE_1.items():
        enc, dec = count(layers, d)
        total = (enc + dec) / 1e6
        print(f"{name:7s} {enc / 1e6:8.1f}M {dec / 1e6:8.1f}M {total:8.1f}M {table:7d}M {100 * (total - table) / table:+6.1f}%")
    layers, d, heads, _ = TABLE_1["Large"]
    print(f"Large head dimension: {d} / {heads} = {d // heads}")
    enc, dec = count(layers, d)
    print(f"Large token embedding: {N_VOCAB * d / 1e6:.1f}M of {(enc + dec) / 1e6:.1f}M")
    print(f"Tiny share in the token embedding: {N_VOCAB * 384 / sum(count(4, 384)) * 100:.1f}%")
