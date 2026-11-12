"""FLOPs-matched exchange between model size and test-time compute (Section 7).

Pretraining costs X = 6 N D_pretrain FLOPs and inference costs Y = 2 N D_inference.
A model M times larger multiplies both by M. The smaller model can match that
total by multiplying its inference compute by M + 3 (D_pretrain / D_inference) (M - 1).
With R = D_inference / D_pretrain, the factor is M + 3 (M - 1) / R.

Run: python3 scripts/flops_matched.py
"""

M = 14                                  # the paper's ~14x larger model
R_VALUES = [0.16, 0.79, 22]            # R << 1, R ~ 1, R >> 1 (Section 7, Figure 9)

# Check the closed form against the definitions with concrete token counts.
n_params = 1.0e9
d_pretrain = 1.0e12
print(f"M = {M}")
print(" R     D_inference   closed form   from X and Y")
for r in R_VALUES:
    d_inf = r * d_pretrain
    x = 6 * n_params * d_pretrain
    y = 2 * n_params * d_inf
    big_total = M * (x + y)                         # M times larger model, greedy decoding
    y_matched = big_total - x                       # what the small model may spend on inference
    from_flops = y_matched / y
    closed = M + 3 * (M - 1) / r
    print(f"{r:5.2f}  {d_inf:.2e}   {closed:11.2f}   {from_flops:11.2f}")

print()
print("Small cases to check by hand")
for m, r in [(2, 1.0), (2, 3.0), (14, 1e9)]:
    print(f"  M={m}, R={r:g}: factor {m + 3 * (m - 1) / r:.4f}")
