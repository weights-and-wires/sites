"""Muon's extra FLOPs and the Muon/AdamW parameter split.

Blog post sections: "Runtime analysis" and "Empirical considerations".
"""


def ns_flops(n, m, steps=5):
    """Matmul FLOPs of the NS iteration for an n x m matrix, m <= n."""
    assert m <= n
    return steps * 2 * (2 * n * m * m + m**3)


if __name__ == "__main__":
    # The two overhead examples in the blog post: T * m / B.
    for name, T, m, B in [("NanoGPT speedrun", 5, 768, 524288), ("Llama 405B", 5, 16384, 16_000_000)]:
        print(f"{name}: T*m/B = {T}*{m}/{B} = {100 * T * m / B:.2f}%")

    # Exact count for one square 768 x 768 matrix against a forward and
    # backward pass over B tokens (6 n m B FLOPs).
    n = m = 768
    B = 524288
    ns = ns_flops(n, m)
    base = 6 * n * m * B
    print(f"\nsquare {n}x{m}: NS FLOPs = {ns:,} ({ns / 1e9:.2f} GFLOPs); 6 T n m^2 bound = {6 * 5 * n * m * m:,}")
    print(f"baseline step FLOPs (6 n m B) = {base:,}; ratio = {100 * ns / base:.2f}%")

    # Tall matrix: an MLP up-projection, 3072 x 768 (n = 3072, m = 768).
    n, m = 3072, 768
    print(f"tall {n}x{m}: NS FLOPs = {ns_flops(n, m):,}; 6 T n m^2 = {6 * 5 * n * m * m:,}")
    print(f"tall {n}x{m}: ratio to baseline = {100 * ns_flops(n, m) / (6 * n * m * 524288):.2f}%")

    # Parameter split for a GPT-2-small-shaped model with Q, K, V separate and no linear biases.
    d, L, V, ctx = 768, 12, 50257, 1024
    hidden = L * (3 * d * d + d * d + d * 4 * d + 4 * d * d)
    emb = V * d + ctx * d
    head = V * d
    vec = L * (2 * 2 * d) + 2 * d  # two LayerNorms (weight and bias) per block, one final LayerNorm, no other biases
    total = hidden + emb + head + vec
    print(f"\nhidden matrices (Muon): {hidden:,} ({100 * hidden / total:.1f}% of {total:,})")
    print(f"embeddings (AdamW): {emb:,}; output head (AdamW): {head:,}; vectors (AdamW): {vec:,}")
