"""Per-GPU model-state memory for data parallelism and ZeRO-DP.

Notation follows Figure 1 and Section 5 of the ZeRO paper: Psi parameters,
K = 12 bytes of optimizer state per parameter (fp32 copy, momentum, variance),
and Nd data-parallel workers. One GB here is 1e9 bytes, as in the paper.
"""

GB = 1e9
K = 12


def model_state_bytes(psi, nd, stage, k=K):
    params, grads, opt = 2 * psi, 2 * psi, k * psi
    if stage >= 1:
        opt /= nd
    if stage >= 2:
        grads /= nd
    if stage >= 3:
        params /= nd
    return params + grads + opt


def main():
    gpt2 = 1.5e9
    print("GPT-2, 1.5B parameters, fp16 weights alone:", 2 * gpt2 / GB, "GB")
    print("GPT-2, 1.5B parameters, mixed-precision Adam:", model_state_bytes(gpt2, 1, 0) / GB, "GB")

    print("\n7B model, bytes per parameter:")
    names = ["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    for nd in (8, 64):
        print(f"  Nd = {nd}")
        for stage, name in enumerate(names):
            b = model_state_bytes(7e9, nd, stage)
            print(f"    {name:7s} {b / 7e9:7.3f} bytes/param  {b / GB:8.2f} GB per GPU")

    print("\nTable 1 row, 7.5B model, Nd = 64:")
    for stage, name in enumerate(names):
        print(f"  {name:7s} {model_state_bytes(7.5e9, 64, stage) / GB:7.2f} GB")

    print("\nLargest model that fits in 32 GB of model states, Nd = 64:")
    for stage in (1, 2, 3):
        coeff = model_state_bytes(1.0, 64, stage)
        print(f"  stage {stage}: {32 * GB / coeff / 1e9:6.1f} B parameters")
    print("  DDP:", round(32 * GB / model_state_bytes(1.0, 64, 0) / 1e9, 2), "B parameters")

    print("\n1T model, Nd = 1024, ZeRO-3:", round(model_state_bytes(1e12, 1024, 3) / GB, 1), "GB")


if __name__ == "__main__":
    main()
