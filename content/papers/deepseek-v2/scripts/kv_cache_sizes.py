"""Compute the KV cache sizes per token for MHA, GQA, MQA, and MLA.

Uses the DeepSeek-V2 hyper-parameters from Section 3.1.2 and the formulas in
Table 1 (Section 2.1.4).

Run: python3 scripts/kv_cache_sizes.py
"""

n_h, d_h, layers = 128, 128, 60
d_c, d_hR = 512, 64
d = 5120

mha = 2 * n_h * d_h * layers
mqa = 2 * d_h * layers
mla = (d_c + d_hR) * layers
print("MHA elements per token   =", mha)
for n_g in (8, 2, 1):
    print(f"GQA n_g={n_g} elements     =", 2 * n_g * d_h * layers)
print("MQA elements per token   =", mqa)
print("MLA elements per token   =", mla)
print("MLA latent only (d_c l)  =", d_c * layers)
print("MHA / MLA                =", round(mha / mla, 2))
print("MLA saves vs MHA         =", f"{100 * (1 - mla / mha):.2f}%")
print("equivalent GQA groups    =", (d_c + d_hR) / (2 * d_h))
print("d_c / d_h, d_hR / d_h    =", d_c / d_h, d_hR / d_h)

ctx = 131072
for name, n in (("MHA", mha), ("MLA", mla)):
    gib = n * 2 * ctx / 2**30
    print(f"{name} bf16 cache, 128K tokens = {gib:.3f} GiB per sequence")

print("\nper-layer projection sizes (weights)")
print("W_DKV (d_c x d)          =", d_c * d)
print("W_UK  (d_h n_h x d_c)    =", d_h * n_h * d_c)
print("W_UV  (d_h n_h x d_c)    =", d_h * n_h * d_c)
print("W_KR  (d_hR x d)         =", d_hR * d)
print("full-rank W_K (d_h n_h x d) =", d_h * n_h * d)

print("\nlow-rank key path")
print("W_DKV + W_UK             =", d_c * d + d_h * n_h * d_c)
print("full-rank / low-rank     =", round(d_h * n_h * d / (d_c * d + d_h * n_h * d_c), 2))
print("rank bound of K          =", d_c, "of", n_h * d_h)

print("\nMoE layer (Section 3.1.2)")
N_s, N_r, K_r = 2, 160, 6
print("experts per layer        =", N_s + N_r)
print("experts used per token   =", N_s + K_r)
