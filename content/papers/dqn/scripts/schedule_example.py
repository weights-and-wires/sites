"""Numbers for the epsilon schedule, frame skipping, and reward clipping (Section 5).

Run: python3 scripts/schedule_example.py
"""

import numpy as np


def epsilon_at(frame, start=1.0, end=0.1, anneal_frames=1_000_000):
    return max(end, start + (end - start) * frame / anneal_frames)


for f in (0, 250_000, 500_000, 1_000_000, 5_000_000):
    print(f"frame {f:>9,}: epsilon = {epsilon_at(f):.3f}")

k = 4
print(f"\nwith frame skip k = {k}:")
print(f"  10,000,000 frames = {10_000_000 // k:,} agent decisions")
print(f"  epsilon reaches 0.1 after {1_000_000 // k:,} decisions")
print(f"  epsilon at the end of the first million frames is {epsilon_at(1_000_000):.1f}")

scores = np.array([-30.0, 0.0, 1.0, 4.0, 200.0])
print("\nreward clipping:", scores.tolist(), "->", np.sign(scores).tolist())
