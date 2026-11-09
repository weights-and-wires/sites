#!/usr/bin/env python3
"""Parameters grow with the number of experts N, but per-token FLOPs do not.

Sizes follow the T5-Base row of Table 9: d_model = 768, d_ff = 2048, and a
GEGLU feed-forward layer, which has three d_model x d_ff matrices (an input
gate, an input projection, and an output projection). The counts cover one
feed-forward layer and ignore biases. Notation: W_r is d_model x N (Section
2.1); each expert is one such FFN.
"""

d_model, d_ff = 768, 2048
ffn_params = 3 * d_model * d_ff
ffn_flops = 2 * ffn_params  # one multiply-add per weight, counted as 2 FLOPs

print(f"dense FFN: params {ffn_params:,}, FLOPs per token {ffn_flops:,}")
for N in (1, 4, 16, 64, 128):
    router = d_model * N
    params = N * ffn_params + router
    router_flops = 2 * router
    flops = ffn_flops + router_flops  # one expert per token (top-1)
    print(f"N={N:>3}: params {params:>13,} ({params / ffn_params:6.2f}x), "
          f"FLOPs per token {flops:>10,} ({flops / ffn_flops:.4f}x), "
          f"router share of FLOPs {router_flops / flops:.4%}")

# top-2 (Shazeer et al., 2017) doubles the expert FLOPs.
N = 128
top2 = 2 * ffn_flops + 2 * d_model * N
print(f"top-2 with N={N}: FLOPs per token {top2:,} ({top2 / ffn_flops:.4f}x of dense)")
