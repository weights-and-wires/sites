"""Rebuild the parameter counts in Table 2 of the LLaMA paper.

Run: python3 scripts/param_count.py

Assumptions that come from the released code, not from the paper text: a
vocabulary of 32,000 tokens, an untied input embedding and output matrix, no
biases, one final RMSNorm, and an FFN width of 2/3 * 4d rounded up to a multiple
of 256.
"""

import math

VOCAB = 32000
# name, d, n_heads, n_layers, Table 2 params (billions)
MODELS = [("7B", 4096, 32, 32, 6.7), ("13B", 5120, 40, 40, 13.0), ("33B", 6656, 52, 60, 32.5), ("65B", 8192, 64, 80, 65.2)]


def hidden(d, multiple_of=256):
    return multiple_of * math.ceil(int(2 * 4 * d / 3) / multiple_of)


def count(d, layers, vocab=VOCAB):
    attn = 4 * d * d
    ffn = 3 * d * hidden(d)
    norms = 2 * d
    per_layer = attn + ffn + norms
    return per_layer, layers * per_layer + 2 * vocab * d + d


for name, d, heads, layers, table in MODELS:
    per_layer, total = count(d, layers)
    print(f"LLaMA-{name}: d = {d}, heads = {heads}, head dim = {d // heads}, layers = {layers}, ffn hidden = {hidden(d)}")
    print(f"  per layer = {per_layer:,}; total = {total:,} = {total / 1e9:.2f}B (Table 2: {table}B)")

per_layer, total = count(4096, 32)
d = 4096
print()
print("LLaMA-7B split:")
print(f"  attention  4 d^2      = {4 * d * d:,}  ({4 * d * d * 32 / total:.1%} of the model)")
print(f"  SwiGLU FFN 3 d h      = {3 * d * hidden(d):,}  ({3 * d * hidden(d) * 32 / total:.1%} of the model)")
print(f"  embeddings 2 V d      = {2 * VOCAB * d:,}  ({2 * VOCAB * d / total:.1%} of the model)")
print("  with tied embeddings the total would be", f"{total - VOCAB * d:,}")
