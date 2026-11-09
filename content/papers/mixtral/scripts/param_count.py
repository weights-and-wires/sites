"""Count total and active parameters and matmul FLOPs for Mixtral 8x7B.

Run: python3 scripts/param_count.py

The architecture values come from Table 1 of the paper. Assumptions that come
from the released code and not from the paper text: untied input embedding and
output matrix, no biases, two RMSNorm gains per block plus one final gain, and a
router of shape (dim, num_experts) with no bias.
"""

MIXTRAL = dict(dim=4096, layers=32, head_dim=128, hidden=14336, heads=32,
               kv_heads=8, vocab=32000, experts=8, top_k=2)
# Llama 2 70B for comparison, from the Llama 2 model card values.
LLAMA2_70B = dict(dim=8192, layers=80, head_dim=128, hidden=28672, heads=64,
                  kv_heads=8, vocab=32000, experts=1, top_k=1)


def counts(c):
    d = c["dim"]
    attn = d * c["heads"] * c["head_dim"] * 2 + d * c["kv_heads"] * c["head_dim"] * 2
    expert = 3 * d * c["hidden"]
    router = d * c["experts"] if c["experts"] > 1 else 0
    norms = 2 * d
    embed = 2 * c["vocab"] * d
    per_layer_total = attn + c["experts"] * expert + router + norms
    per_layer_active = attn + c["top_k"] * expert + router + norms
    total = c["layers"] * per_layer_total + embed + d
    active = c["layers"] * per_layer_active + embed + d
    return dict(attn=attn, expert=expert, router=router, embed=embed,
                total=total, active=active)


m = counts(MIXTRAL)
dense = counts(dict(MIXTRAL, experts=1, top_k=1))
l70 = counts(LLAMA2_70B)

print("Mixtral 8x7B, per block")
print("  attention weights   =", f"{m['attn']:,}")
print("  one expert (SwiGLU) =", f"{m['expert']:,}")
print("  router              =", f"{m['router']:,}")
print("Mixtral 8x7B, whole model")
print("  embeddings (in+out) =", f"{m['embed']:,}")
print("  total parameters    =", f"{m['total']:,}", f"= {m['total'] / 1e9:.2f}B")
print("  active parameters   =", f"{m['active']:,}", f"= {m['active'] / 1e9:.2f}B")
print("  active / total      =", f"{m['active'] / m['total']:.4f}")
print("  experts' share of total =",
      f"{MIXTRAL['layers'] * MIXTRAL['experts'] * m['expert'] / m['total']:.4f}")
print("One expert only (the dense Mistral 7B shape)")
print("  parameters          =", f"{dense['total']:,}", f"= {dense['total'] / 1e9:.2f}B")
print("Llama 2 70B")
print("  parameters          =", f"{l70['total']:,}", f"= {l70['total'] / 1e9:.2f}B")
print("  Llama 2 70B / Mixtral active =", f"{l70['total'] / m['active']:.2f}")
print("Forward matmul FLOPs per token, 2 x active parameters (no attention-score term)")
print("  Mixtral             =", f"{2 * m['active'] / 1e9:.2f} GFLOPs")
print("  Llama 2 70B         =", f"{2 * l70['total'] / 1e9:.2f} GFLOPs")
print("  one-expert shape    =", f"{2 * dense['total'] / 1e9:.2f} GFLOPs")
print("Weight memory at 2 bytes per parameter (bfloat16)")
print("  Mixtral             =", f"{2 * m['total'] / 1e9:.1f} GB")
print("  Llama 2 70B         =", f"{2 * l70['total'] / 1e9:.1f} GB")
