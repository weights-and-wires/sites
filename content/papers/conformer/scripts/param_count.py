#!/usr/bin/env python3
"""Parameter counts for Conformer S, M, L (Table 1) and the WER ratios."""

# Table 1: name, d, layers, heads, kernel, decoder dim, reported params (M).
CONFIGS = [
    ("S", 144, 16, 4, 32, 320, 10.3),
    ("M", 256, 16, 4, 32, 640, 30.7),
    ("L", 512, 17, 8, 32, 640, 118.8),
]


def ffn_params(d, mult=4):
    return 2 * d + (d * mult * d + mult * d) + (mult * d * d + d)  # LN, up, down


def mhsa_params(d):
    # LN, Q K V O with biases, positional projection without bias, u and v.
    return 2 * d + 4 * (d * d + d) + d * d + 2 * d


def conv_params(d, k):
    # LN, pointwise d->2d, depthwise, BN scale and shift, pointwise d->d.
    return 2 * d + (d * 2 * d + 2 * d) + (d * k + d) + 2 * d + (d * d + d)


def block_params(d, k):
    return 2 * ffn_params(d) + mhsa_params(d) + conv_params(d, k) + 2 * d  # final LN


for name, d, layers, heads, k, dec, total in CONFIGS:
    b = block_params(d, k)
    enc = b * layers / 1e6
    print(f"{name}: d={d} blocks={layers}")
    print(f"   ffn {ffn_params(d)}  mhsa {mhsa_params(d)}  conv {conv_params(d, k)}  block {b}")
    print(f"   encoder blocks {enc:.2f}M of {total}M reported; rest {total - enc:.2f}M ({100 * (total - enc) / total:.0f}%)")
    print(f"   depthwise share of one conv module: {d * k + d} of {conv_params(d, k)}")

print("Heads: per-head dim", [d // h for _, d, _, h, _, _, _ in CONFIGS])

# Headline comparisons, with numbers from Table 2.
print("relative gain over Transformer Transducer, test-other with LM: %.1f%%" % (100 * (4.6 - 3.9) / 4.6))
print("params, Conformer M vs Transformer Transducer: %.1f%%" % (100 * 30.7 / 139))
