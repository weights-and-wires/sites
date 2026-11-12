"""Roofline cost of one iteration for the linear layers (Section 3.1).

Run from the paper folder: python3 scripts/roofline_cost.py

The model is deliberately small. It counts only the weight traffic and the
matmul FLOPs of the linear layers, which the paper says dominate the runtime
(Section 3.1, Figure 4). It ignores attention, activations, and fixed
overheads. The hardware numbers are the public A100 80GB datasheet values, and
the parameter count is Mistral-7B's. Neither comes from the paper.
"""

PEAK = 312e12  # A100 fp16 tensor-core peak, FLOP/s (datasheet)
BW = 2.039e12  # A100 80GB HBM bandwidth, bytes/s (datasheet)
PARAMS = 7.24e9  # Mistral-7B parameters
BYTES = 2  # fp16 bytes per parameter


def intensity(n_tokens, bytes_per_param=BYTES):
    """FLOPs per byte for a pass of n_tokens through the weights."""
    return 2 * n_tokens / bytes_per_param


def iteration_time(n_tokens):
    t_math = 2 * PARAMS * n_tokens / PEAK
    t_mem = BYTES * PARAMS / BW
    return max(t_math, t_mem), ("memory" if t_mem > t_math else "compute")


if __name__ == "__main__":
    ridge_intensity = PEAK / BW
    ridge_tokens = ridge_intensity * BYTES / 2
    print(f"device ridge point: {ridge_intensity:.1f} FLOP/byte")
    print(f"tokens at the ridge: {ridge_tokens:.1f}")
    t_dec, _ = iteration_time(1)
    print(f"memory floor (weight read): {t_dec * 1e3:.2f} ms")
    print()
    print("tokens  intensity  time_ms  bound    tokens/s")
    for n in (1, 32, 128, 153, 256, 512, 2048, 4096):
        t, b = iteration_time(n)
        print(f"{n:6d}  {intensity(n):9.0f}  {t * 1e3:7.2f}  {b:7s}  {n / t:8.0f}")
    print()
    slo = 5 * t_dec
    print(f"strict-SLO style bound, 5 x memory floor = {slo * 1e3:.2f} ms")
    budget = int(slo * PEAK / (2 * PARAMS))
    print(f"largest budget under that bound: {budget} tokens")
    print()
    d = 32
    print(f"decode batch of {d} alone: {iteration_time(d)[0] * 1e3:.2f} ms")
    for tau in (153, 512):
        t, b = iteration_time(tau)
        print(f"hybrid batch at budget {tau}: {d} decodes + {tau - d} prefill tokens "
              f"-> {t * 1e3:.2f} ms ({t / t_dec:.2f}x the floor, {b}-bound)")
