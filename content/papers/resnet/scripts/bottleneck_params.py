"""Compute the parameter counts for the basic and bottleneck blocks.

Run: python3 scripts/bottleneck_params.py
"""


def conv(k, c_in, c_out):
    return k * k * c_in * c_out


basic = 2 * conv(3, 64, 64)
bott = conv(1, 256, 64) + conv(3, 64, 64) + conv(1, 64, 256)
proj = conv(1, 256, 256)
print("Figure 5 blocks on 56x56 feature maps (no biases, no BN)")
print(f"basic block, two 3x3 at 64:        {basic:,}")
print(f"bottleneck, 256 -> 64 -> 64 -> 256: {bott:,}")
print(f"  1x1 reduce {conv(1, 256, 64):,}; 3x3 {conv(3, 64, 64):,}; 1x1 restore {conv(1, 64, 256):,}")
print(f"multiply-adds at 56x56 (weights x 3136): basic {basic * 56 * 56:,}; bottleneck {bott * 56 * 56:,}")
print()
print("Projection shortcut on the bottleneck")
print(f"extra 1x1 256 -> 256: {proj:,}")
print(f"block with projection: {bott + proj:,}, ratio {(bott + proj) / bott:.3f}")
print()
print("Same ends, but a basic block at 256 width")
b256 = 2 * conv(3, 256, 256)
print(f"basic block at 256: {b256:,}, which is {b256 / bott:.1f} times the bottleneck")
