"""Compute the numbers for the crop-and-concatenate example.

Run: python3 content/papers/unet/scripts/skip_example.py
"""

import numpy as np

skip = np.arange(36).reshape(1, 6, 6)  # a 1-channel 6 x 6 contracting-path map
up = np.full((1, 4, 4), -1)  # a 1-channel 4 x 4 up-convolution output
top = (skip.shape[1] - up.shape[1]) // 2
left = (skip.shape[2] - up.shape[2]) // 2
cropped = skip[:, top : top + 4, left : left + 4]
print("crop offset per side:", top)
print("cropped skip rows:")
for row in cropped[0]:
    print(" ", row.tolist())
merged = np.concatenate([cropped, up], axis=0)
print("concatenated shape:", merged.shape)
corner = skip[:, :4, :4]
print("top-left crop (bug) first row:", corner[0, 0].tolist())

# Channels at the last expansion step, with and without the concatenation.
c = 64
print("channels after concatenation:", c + c)
print("weights in the 3x3 conv after concat:", 9 * (2 * c) * c, " without skip:", 9 * c * c)
