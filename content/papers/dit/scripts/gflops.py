"""Count forward-pass Gflops for DiT from the Table 1 configs.

Counts one multiply-add as one flop, the convention under which the counts match
Table 4 of the paper. Run: python3 scripts/gflops.py
"""

CONFIGS = {"S": (12, 384), "B": (12, 768), "L": (24, 1024), "XL": (28, 1152)}
# Table 4 values, Gflops at I = 32, C = 4.
TABLE4 = {
    ("S", 8): 0.36, ("S", 4): 1.41, ("S", 2): 6.06,
    ("B", 8): 1.42, ("B", 4): 5.56, ("B", 2): 23.01,
    ("L", 8): 5.01, ("L", 4): 19.70, ("L", 2): 80.71,
    ("XL", 8): 7.39, ("XL", 4): 29.05, ("XL", 2): 118.64,
}


def tokens(I, p):
    return (I // p) ** 2


def block_macs(T, d):
    qkv_proj = 4 * T * d * d       # Q, K, V and output projections
    attn = 2 * T * T * d           # QK^T and softmax(.)V
    mlp = 8 * T * d * d            # d -> 4d -> d
    adaln = 6 * d * d              # SiLU + Linear(d, 6d) on one conditioning vector
    return qkv_proj + attn + mlp + adaln


def dit_gflops(name, p, I=32, C=4):
    N, d = CONFIGS[name]
    T = tokens(I, p)
    patchify = T * (p * p * C) * d
    blocks = N * block_macs(T, d)
    final = 2 * d * d + T * d * (p * p * 2 * C)  # adaLN (shift, scale) + linear decoder
    return (patchify + blocks + final) / 1e9


if __name__ == "__main__":
    print("config      T   computed  Table 4  error")
    for (name, p), want in TABLE4.items():
        got = dit_gflops(name, p)
        print(f"DiT-{name}/{p:<2}  {tokens(32, p):4d}  {got:8.2f}  {want:7.2f}  {100 * (got - want) / want:+5.1f}%")
    print()
    print("XL/2 at 512x512 (I = 64):", round(dit_gflops("XL", 2, I=64), 1), "Gflops (paper: 524.6)")
    print("Halving p quadruples T:", tokens(32, 4) / tokens(32, 8), "; XL Gflops ratio /2 vs /4 =",
          round(dit_gflops("XL", 2) / dit_gflops("XL", 4), 2))
    print()
    # Parameters per block, to show patch size barely changes them.
    for name in CONFIGS:
        N, d = CONFIGS[name]
        per_block = 12 * d * d + 6 * d * d
        print(f"DiT-{name}: ~{N * per_block / 1e6:.0f}M block parameters (weights only)")
