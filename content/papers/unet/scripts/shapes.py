"""Compute the feature-map sizes, crops, and parameter count of the U-Net in Figure 1.

Run: python3 content/papers/unet/scripts/shapes.py
"""

import math


def trace(n, levels=4, base=64):
    """Return a list of (label, size, channels) for the paper's architecture."""
    rows = [("input", n, 1)]
    skips = []
    c = base
    for lvl in range(levels):
        n -= 2
        rows.append((f"down{lvl} conv1", n, c))
        n -= 2
        rows.append((f"down{lvl} conv2", n, c))
        skips.append(n)
        if n % 2:
            return rows, None
        n //= 2
        rows.append((f"down{lvl} pool", n, c))
        c *= 2
    n -= 2
    rows.append(("bottom conv1", n, c))
    n -= 2
    rows.append(("bottom conv2", n, c))
    crops = []
    for lvl in reversed(range(levels)):
        c //= 2
        n *= 2
        rows.append((f"up{lvl} up-conv", n, c))
        crops.append((lvl, skips[lvl], n))
        n -= 2
        rows.append((f"up{lvl} conv1", n, c))
        n -= 2
        rows.append((f"up{lvl} conv2", n, c))
    rows.append(("1x1 conv", n, 2))
    return rows, crops


rows, crops = trace(572)
print("Figure 1 trace for a 572 x 572 input:")
for label, n, c in rows:
    print(f"  {label:<14} {n} x {n}  ({c} channels)")
print("crops (skip size -> up-conv size, border removed per side):")
for lvl, skip, up in crops:
    print(f"  level {lvl}: {skip} -> {up}, {(skip - up) // 2} pixels per side")

# Layer and parameter counts, with biases, for 1 input channel and 2 output classes.
convs3 = 0
params = 0
cin = 1
c = 64
for lvl in range(4):
    for _ in range(2):
        params += 9 * cin * c + c
        convs3 += 1
        cin = c
    c *= 2
for _ in range(2):  # bottom
    params += 9 * cin * c + c
    convs3 += 1
    cin = c
upconvs = 0
for lvl in reversed(range(4)):
    half = c // 2
    params += 4 * c * half + half  # 2x2 up-convolution halves the channels
    upconvs += 1
    c = half
    params += 9 * (2 * c) * c + c  # after concatenation: 2c in, c out
    params += 9 * c * c + c
    convs3 += 2
params += 1 * 1 * 64 * 2 + 2
print(f"3x3 convs: {convs3}, 2x2 up-convs: {upconvs}, 1x1 convs: 1, total: {convs3 + upconvs + 1}")
print(f"parameters (with biases, 1 input channel, 2 classes): {params:,}")

# Overlap-tile arithmetic for a 512 x 512 image (Figure 2).
tile_in, tile_out = 572, 388
border = (tile_in - tile_out) // 2
tiles = math.ceil(512 / tile_out)
print(f"border per side: {border}; tiles per axis for 512 px: {tiles}; total tiles: {tiles * tiles}")
print(f"padded image needed: {512 + 2 * border}")

# Valid input sizes for 4 levels between 560 and 600.
ok = [n for n in range(560, 601) if trace(n)[1] is not None and trace(n)[0][-1][1] > 0]
print("valid input sizes in 560..600:", ok)
print("their output sizes:", [trace(n)[0][-1][1] for n in ok])
