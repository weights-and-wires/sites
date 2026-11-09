def dit_gflops(N, d, p, I=32, C=4):
    """Forward-pass Gflops (multiply-adds / 1e9) of a DiT with N blocks and width d.

    I is the latent side length, C its channel count, p the patch size.
    """
    T = (I // p) ** 2
    patchify = T * (p * p * C) * d
    per_block = 4 * T * d * d + 2 * T * T * d + 8 * T * d * d + 6 * d * d
    final = 2 * d * d + T * d * (p * p * 2 * C)
    return (patchify + N * per_block + final) / 1e9
