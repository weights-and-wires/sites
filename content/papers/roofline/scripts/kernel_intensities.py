#!/usr/bin/env python3
"""Operational intensity of matmul, softmax, and LayerNorm.

The paper analyzes SpMV, LBMHD, a stencil, and an FFT. These three kernels are
this page's extension of the same counting method. Counting rules:
  - Bytes are compulsory DRAM traffic: every input read once, every output written once.
  - Matmul counts one multiply and one add per term, so 2*m*k*n FLOPs.
  - Softmax counts 5 FLOPs per element (max compare, subtract, exp, add, divide).
  - LayerNorm counts 8 FLOPs per element (add for the mean; subtract, multiply,
    and add for the variance; subtract, multiply, and scale-and-shift for the output).
  Other conventions give other counts; the conclusion (matmul far above the
  ridge point, softmax and LayerNorm far below) does not change.
"""


def matmul_intensity(m, k, n, b):
    flops = 2 * m * k * n
    traffic = b * (m * k + k * n + m * n)
    return flops, traffic, flops / traffic


def softmax_intensity(rows, cols, b):
    flops = 5 * rows * cols
    traffic = b * 2 * rows * cols
    return flops, traffic, flops / traffic


def layernorm_intensity(rows, cols, b):
    flops = 8 * rows * cols
    traffic = b * (2 * rows * cols + 2 * cols)  # x in, y out, gamma and beta in
    return flops, traffic, flops / traffic


def tiled_matmul_intensity(n, tile, b):
    flops = 2 * n ** 3
    traffic = b * (2 * n ** 3 / tile + n ** 2)
    return flops / traffic


if __name__ == "__main__":
    for b, label in ((4, "FP32"), (2, "BF16")):
        print(label, "(%d bytes per element)" % b)
        for n in (64, 512, 4096):
            f, t, i = matmul_intensity(n, n, n, b)
            print("  matmul %5d^3     FLOPs %.3e  bytes %.3e  intensity %8.2f" % (n, f, t, i))
        f, t, i = softmax_intensity(4096, 4096, b)
        print("  softmax 4096x4096  FLOPs %.3e  bytes %.3e  intensity %8.3f" % (f, t, i))
        f, t, i = layernorm_intensity(4096, 4096, b)
        print("  layernorm 4096x4096 FLOPs %.3e  bytes %.3e  intensity %8.3f" % (f, t, i))
    print("Tiled matmul, n = 4096, BF16")
    for tile in (1, 16, 64, 128, 4096):
        print("  tile %5d -> intensity %8.2f" % (tile, tiled_matmul_intensity(4096, tile, 2)))
    # Smallest tile that reaches a ridge point (the H100 BF16 ridge from gpu_rooflines.py).
    ridge = 989e3 / 3350.0
    n = 8192
    t = next(t for t in range(1, n + 1) if tiled_matmul_intensity(n, t, 2) >= ridge)
    print("Smallest tile reaching the H100 BF16 ridge (%.1f) at n = %d: %d" % (ridge, n, t))
    print("  intensity at tile %d: %.2f; at tile %d: %.2f" % (t - 1, tiled_matmul_intensity(n, t - 1, 2), t, tiled_matmul_intensity(n, t, 2)))
