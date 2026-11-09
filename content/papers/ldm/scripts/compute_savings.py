#!/usr/bin/env python3
"""Cost of pixel-space versus latent-space diffusion for one 512 x 512 RGB image.

The latent channel counts c come from the paper's Table 13 (z-shape row), where
the same U-Net size is used for every downsampling factor f.
"""

H = W = 512
# f -> c, from Table 13 (z-shape: 256x256x3, 128x128x2, 64x64x3, 32x32x4, 16x16x8, 8x8x32)
CHANNELS = {1: 3, 2: 2, 4: 3, 8: 4, 16: 8, 32: 32}

pixel_positions = H * W
pixel_values = H * W * 3
pixel_attn = pixel_positions**2

print(f"image {H}x{W}x3: {pixel_positions:,} positions, {pixel_values:,} values")
print(f"{'f':>3} {'z shape':>12} {'positions':>10} {'values':>10} {'pos ratio':>10} {'value ratio':>12} {'attn ratio':>11}")
for f, c in CHANNELS.items():
    h, w = H // f, W // f
    pos = h * w
    vals = h * w * c
    print(f"{f:>3} {f'{h}x{w}x{c}':>12} {pos:>10,} {vals:>10,} {pixel_positions / pos:>9.0f}x "
          f"{pixel_values / vals:>11.1f}x {pixel_attn / pos**2:>10.0f}x")

# A 1000-step sampler touches every position at every step.
steps = 1000
print()
print(f"positions processed over {steps} steps, f=1: {steps * pixel_positions:,}")
print(f"positions processed over {steps} steps, f=8: {steps * (H // 8) * (W // 8):,}")
print(f"f=8 latent is {pixel_values // (64 * 64 * 4)} times smaller than the RGB image in values")
