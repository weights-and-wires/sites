#!/usr/bin/env python3
"""Rooflines for an A100 and an H100 from datasheet numbers, plus kernel placement.

Spec numbers (dense, no sparsity), from the NVIDIA A100 and H100 datasheets:
  A100 SXM 80 GB: 312 TFLOP/s BF16 tensor core, 19.5 TFLOP/s FP32, 2039 GB/s HBM2e.
  H100 SXM:       989 TFLOP/s BF16 tensor core, 67 TFLOP/s FP32, 3350 GB/s HBM3.
Pass --plot PATH to save a log-log plot (needs matplotlib; the numbers need only Python).
"""
import sys
from kernel_intensities import layernorm_intensity, matmul_intensity, softmax_intensity

GPUS = {
    "A100": {"bf16": 312e3, "fp32": 19.5e3, "bw": 2039.0},  # GFLOP/s, GB/s
    "H100": {"bf16": 989e3, "fp32": 67e3, "bw": 3350.0},
}


def attainable(oi, peak, bw):
    return min(peak, bw * oi)


kernels = {
    "matmul 4096^3 BF16": matmul_intensity(4096, 4096, 4096, 2)[2],
    "matmul 512^3 BF16": matmul_intensity(512, 512, 512, 2)[2],
    "softmax 4096x4096 BF16": softmax_intensity(4096, 4096, 2)[2],
    "layernorm 4096x4096 BF16": layernorm_intensity(4096, 4096, 2)[2],
}

for name, g in GPUS.items():
    print("%s: BF16 peak %.0f TFLOP/s, bandwidth %.0f GB/s" % (name, g["bf16"] / 1e3, g["bw"]))
    print("  ridge point (BF16 tensor): %.1f FLOPs/byte" % (g["bf16"] / g["bw"]))
    print("  ridge point (FP32 CUDA):   %.1f FLOPs/byte" % (g["fp32"] / g["bw"]))
    for k, oi in kernels.items():
        p = attainable(oi, g["bf16"], g["bw"])
        regime = "memory-bound" if oi * g["bw"] < g["bf16"] else "compute-bound"
        print("  %-26s intensity %8.3f -> %9.1f GFLOP/s (%4.1f%% of peak, %s)" % (k, oi, p, 100 * p / g["bf16"], regime))

a, h = GPUS["A100"], GPUS["H100"]
print("H100 / A100: peak x%.2f, bandwidth x%.2f, ridge x%.2f" % (h["bf16"] / a["bf16"], h["bw"] / a["bw"], (h["bf16"] / h["bw"]) / (a["bf16"] / a["bw"])))

if "--plot" in sys.argv:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oi = np.logspace(-1, 4, 200)
    for name, g in GPUS.items():
        plt.loglog(oi, np.minimum(g["bf16"], g["bw"] * oi) / 1e3, label=name)
    for k, v in kernels.items():
        plt.axvline(v, ls=":", lw=0.8)
    plt.xlabel("FLOPs per byte of DRAM traffic")
    plt.ylabel("TFLOP/s")
    plt.legend()
    plt.savefig(sys.argv[sys.argv.index("--plot") + 1], dpi=150)
