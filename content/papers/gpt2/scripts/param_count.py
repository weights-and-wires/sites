"""Parameter counts for the four GPT-2 sizes in Table 2 (Section 2.3).

Assumptions, not stated in the paper's text: the feed-forward width is
4 * d_model, every linear layer has a bias, the input embedding and the output
projection share one matrix, and positions are learned (context 1024).

Run: python3 scripts/param_count.py
"""

V, CTX = 50257, 1024
SIZES = [("117M", 12, 768), ("345M", 24, 1024), ("762M", 36, 1280), ("1542M", 48, 1600)]


def count(layers, d):
    tok = V * d
    pos = CTX * d
    attn = (d * 3 * d + 3 * d) + (d * d + d)  # fused QKV and output projection
    ffn = (d * 4 * d + 4 * d) + (4 * d * d + d)
    ln = 2 * (2 * d)  # two LayerNorms, each with gain and bias
    block = attn + ffn + ln
    final_ln = 2 * d
    return tok, pos, block, tok + pos + layers * block + final_ln


for name, layers, d in SIZES:
    tok, pos, block, total = count(layers, d)
    print(f"{name:>6}: layers={layers:>2} d_model={d:>4} block={block:>10,} total={total:>13,} ({total / 1e6:.1f}M)")

tok, pos, block, total = count(12, 768)
print(f"\nsmallest model: token embedding {tok:,}, positions {pos:,}, one block {block:,}")
print(f"12 blocks = {12 * block:,}; embeddings are {100 * (tok + pos) / total:.1f}% of the total")
print(f"without weight tying the total would be {total + tok:,}")
