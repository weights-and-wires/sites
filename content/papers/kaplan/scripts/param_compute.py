"""Parameter count and training compute for GPT-2-style models (Section 2.1).

Run: python3 scripts/param_compute.py
"""

PF_DAY = 1e15 * 24 * 3600  # 8.64e19 FLOPs, Section 1.3
N_VOCAB, N_CTX = 50257, 1024


def non_embedding(n_layer, d_model):
    return 12 * n_layer * d_model**2  # Equation 2.1 with d_attn = d_ff / 4 = d_model


def embedding(d_model):
    return (N_VOCAB + N_CTX) * d_model  # Table 1, "Embed" row


for name, n_layer, d_model in [("GPT-2 small", 12, 768), ("GPT-2 XL", 48, 1600)]:
    n = non_embedding(n_layer, d_model)
    e = embedding(d_model)
    print(f"{name}: n_layer={n_layer} d_model={d_model}")
    print(f"  N (non-embedding) = {n:.4e}")
    print(f"  embedding params  = {e:.4e}  ({e / (n + e):.1%} of all parameters)")

# Training compute C = 6 N B S (Section 1.3), with the default run of Section 2.2:
# 2.5e5 steps, batch of 512 sequences of 1024 tokens.
n = non_embedding(48, 1600)
B = 512 * 1024
S = 2.5e5
C = 6 * n * B * S
print()
print(f"B = {B} tokens, S = {S:.1e} steps, D = B*S = {B * S:.4e} tokens")
print(f"C = 6*N*B*S = {C:.4e} FLOPs = {C / PF_DAY:.2f} PF-days")
print(f"C per token = 6N = {6 * n:.4e} FLOPs")

# The context-dependent term of Equation 2.2 relative to 2N.
d_attn = 1600
extra = 2 * 48 * N_CTX * d_attn
print(f"forward: 2N = {2 * n:.4e}, context term 2*n_layer*n_ctx*d_attn = {extra:.4e} ({extra / (2 * n):.1%} of 2N)")
