"""Numbers for the "frame stacking" and "architecture" parts of the page.

Run: python3 scripts/frames_and_network.py
"""

# --- Perceptual aliasing -------------------------------------------------
# A ball moves one cell per frame along a row of 5 cells (0..4) and reverses at
# the ends. One frame shows only the ball's cell. Two stacked frames also show
# which way it moved.
def positions(start, direction, n):
    pos, d, out = start, direction, []
    for _ in range(n):
        out.append(pos)
        if not 0 <= pos + d <= 4:
            d = -d
        pos += d
    return out


next_given_one = {}
next_given_two = {}
for start in range(5):
    for d in (-1, 1):
        ps = positions(start, d, 12)
        for t in range(1, len(ps) - 1):
            next_given_one.setdefault(ps[t], set()).add(ps[t + 1])
            next_given_two.setdefault((ps[t - 1], ps[t]), set()).add(ps[t + 1])
print("one frame, ball at cell 2: next cell is one of", sorted(next_given_one[2]))
print("two frames, 1 -> 2: next cell is", sorted(next_given_two[(1, 2)]))
print("two frames, 3 -> 2: next cell is", sorted(next_given_two[(3, 2)]))


# --- Network shapes and parameters (Section 4.1) ------------------------
def conv_out(n, k, s):
    return (n - k) // s + 1  # no padding


h1 = conv_out(84, 8, 4)
h2 = conv_out(h1, 4, 2)
flat = h2 * h2 * 32
print(f"\nconv1: 84x84x4 -> {h1}x{h1}x16   params = {8 * 8 * 4 * 16 + 16}")
print(f"conv2: {h1}x{h1}x16 -> {h2}x{h2}x32   params = {4 * 4 * 16 * 32 + 32}")
print(f"flatten: {flat} units")
print(f"fc1: {flat} -> 256   params = {flat * 256 + 256}")
for K in (4, 18):
    fc2 = 256 * K + K
    total = (8 * 8 * 4 * 16 + 16) + (4 * 4 * 16 * 32 + 32) + (flat * 256 + 256) + fc2
    print(f"output K={K:2d}: 256 -> {K}   params = {fc2}   network total = {total}")

# --- Input size ----------------------------------------------------------
print(f"\nraw frame values : {210 * 160 * 3}   (210 x 160 x 3)")
print(f"network input    : {84 * 84 * 4}   (84 x 84 x 4)")
