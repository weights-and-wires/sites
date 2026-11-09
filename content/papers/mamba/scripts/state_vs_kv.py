"""Compute the memory numbers: the KV cache against the Mamba state, and the
size of the expanded state that the fused scan never writes to HBM.

The Transformer sizes come from Table 12 of the paper (the 1.3B model:
24 layers, d_model 2048, 32 heads of width 64). The Mamba sizes are my
assumptions for a model of the same width: 48 layers (two Mamba blocks per
Transformer layer, Section 3.4), expansion E = 2, state size N = 16
(Section 4.5), and a causal convolution of width 4 (the reference code's
default; the paper doesn't state it).

Run: python3 scripts/state_vs_kv.py
"""


def kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, bytes_per_elem=2):
    return 2 * n_layers * n_heads * head_dim * seq_len * bytes_per_elem


def mamba_state_bytes(n_layers, d_model, expand=2, d_state=16, d_conv=4, bytes_per_elem=2):
    d_inner = expand * d_model
    return n_layers * (d_inner * d_state + d_inner * (d_conv - 1)) * bytes_per_elem


kv_layers, heads, head_dim = 24, 32, 64
m_layers, d_model = 48, 2048

state = mamba_state_bytes(m_layers, d_model)
per_token = kv_cache_bytes(kv_layers, heads, head_dim, 1)
print(f"KV cache per token : {per_token:,} bytes")
print(f"Mamba state (fixed): {state:,} bytes")
print()
print("tokens      KV cache (bytes)   KV / Mamba state")
for T in [1_000, 10_000, 100_000]:
    kv = kv_cache_bytes(kv_layers, heads, head_dim, T)
    print(f"{T:<11,} {kv:<18,} {kv / state:,.1f}")
print()
print(f"At 100,000 tokens: KV cache = {kv_cache_bytes(kv_layers, heads, head_dim, 100_000) / 1e9:.2f} GB, "
      f"Mamba state = {state / 1e6:.2f} MB")
print("Largest length where the cache is no larger than the state:", state // per_token)
print("Smallest length where the cache is larger:", state // per_token + 1)

# Section 3.3.2: the expanded state h has shape (B, L, D, N), N times the input x.
B, L, D, N = 1, 2048, 2 * d_model, 16
print()
print(f"x has {B * L * D:,} elements; the expanded h has {B * L * D * N:,} elements ({N}x larger)")
