"""Parameter and training-compute counts for the eight models in Table 2.1.

Parameters follow the GPT-2 layout the paper says it reuses (Section 2.1):
d_ff = 4 * d_model, learned positions, biases on every linear layer, one
embedding matrix shared with the output projection, and the 50,257-token GPT-2
vocabulary. The paper states the 4x feed-forward width, but not the other
choices. Training compute is Appendix D's 6 * N * D, with attention ignored.
Table D.1 values are copied from the paper for comparison.

Run: python3 scripts/model_size.py
"""

V, N_CTX = 50257, 2048
TOKENS = 300e9  # Table 2.1: all models train for 300B tokens
PF_DAY = 8.64e19  # Appendix D: flops in one petaflop/s-day

# name, n_layers, d_model, (n_params in M, total train flops, PF-days) from Table D.1
MODELS = [
    ("Small", 12, 768, 125, 2.25e20, 2.60),
    ("Medium", 24, 1024, 356, 6.41e20, 7.42),
    ("Large", 24, 1536, 760, 1.37e21, 15.8),
    ("XL", 24, 2048, 1320, 2.38e21, 27.5),
    ("2.7B", 32, 2560, 2650, 4.77e21, 55.2),
    ("6.7B", 32, 4096, 6660, 1.20e22, 139),
    ("13B", 40, 5140, 12850, 2.31e22, 268),  # Table 2.1 prints d_model = 5140
    ("175B", 96, 12288, 174600, 3.14e23, 3640),
]


def block_params(d):
    attn = d * 3 * d + 3 * d + d * d + d
    ffn = d * 4 * d + 4 * d + 4 * d * d + d
    ln = 2 * (2 * d)
    return attn + ffn + ln  # = 12 d^2 + 13 d


def n_params(layers, d):
    return V * d + N_CTX * d + layers * block_params(d) + 2 * d


def attention_fraction(d):
    """Share of forward flops spent on QK^T and AV at full context, dense attention."""
    return N_CTX / (6 * d + N_CTX)


print(f"{'model':<7}{'layers':>7}{'d_model':>8}{'params (mine)':>16}{'Table D.1':>16}{'diff':>8}{'6ND flops':>12}{'PF-days':>9}{'table PF':>9}")
for name, L, d, p_tab, f_tab, pf_tab in MODELS:
    p = n_params(L, d)
    flops = 6 * p * TOKENS
    print(f"{name:<7}{L:>7}{d:>8}{p:>16,}{p_tab * 1e6:>16,.0f}{100 * (p / (p_tab * 1e6) - 1):>7.2f}%{flops:>12.3e}{flops / PF_DAY:>9.1f}{pf_tab:>9}")

print()
print("one block at d_model = 12288:", f"{block_params(12288):,}")
print("12 * L * d^2 for 175B:", f"{12 * 96 * 12288**2:,}", "of", f"{n_params(96, 12288):,}")
print("embeddings (token + position) for 175B:", f"{V * 12288 + N_CTX * 12288:,}")
print()
print("13B with d_model = 5120 (40 heads x d_head 128):", f"{n_params(40, 5120):,}", "vs", f"{n_params(40, 5140):,}", "at 5140")
print("40 * 128 =", 40 * 128, "so d_model = 5140 is not heads x d_head")
print()
p175 = 174.6e9
print(f"Table D.1 params with 6ND: {6 * p175 * TOKENS:.3e} flops = {6 * p175 * TOKENS / PF_DAY:.0f} PF-days")
print(f"tokens per parameter, 175B: {TOKENS / p175:.2f}")
print()
for name, L, d, *_ in (MODELS[0], MODELS[3], MODELS[7]):
    print(f"attention matmul share of forward flops, {name}: {100 * attention_fraction(d):.1f}% (full context, dense)")
