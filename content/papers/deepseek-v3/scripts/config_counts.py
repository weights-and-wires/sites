"""Counts implied by the DeepSeek-V3 hyper-parameters (Section 4.2). Estimates, not the paper's own numbers."""

D, LAYERS, DENSE = 7168, 61, 3
N_S, N_R, K_R, D_E = 1, 256, 8, 2048
N_H, D_H, D_C, D_HR = 128, 128, 512, 64


def main():
    moe_layers = LAYERS - DENSE
    per_expert = 3 * D * D_E  # SwiGLU: gate, up, down
    print(f"MoE layers = {LAYERS} - {DENSE} = {moe_layers}")
    print(f"params per expert (3 x {D} x {D_E}) = {per_expert:,}")
    total = moe_layers * (N_S + N_R) * per_expert
    active = moe_layers * (N_S + K_R) * per_expert
    print(f"expert params, all: {moe_layers} x {N_S + N_R} x expert = {total / 1e9:.1f}B (paper total: 671B)")
    print(f"expert params, active: {moe_layers} x {N_S + K_R} x expert = {active / 1e9:.1f}B (paper active: 37B)")
    print(f"experts used per token per layer: {N_S + K_R} of {N_S + N_R}")
    mha = 2 * N_H * D_H * LAYERS
    mla = (D_C + D_HR) * LAYERS
    print(f"KV cache per token: MHA {mha:,} elements, MLA {mla:,} elements, ratio {mha / mla:.1f}")


if __name__ == "__main__":
    main()
